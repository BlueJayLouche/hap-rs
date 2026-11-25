//! wgpu Integration for HAP Video Playback and Encoding
//!
//! Provides GPU-accelerated HAP video playback using wgpu.
//! Uploads compressed DXT textures directly to the GPU without CPU decompression.
//!
//! Also provides high-level video encoding APIs for creating HAP videos.
//!
//! # Required wgpu Features
//!
//! HAP uses BC (Block Compression) texture formats, which require the
//! `TEXTURE_COMPRESSION_BC` feature to be enabled on the wgpu device.
//!
//! ```no_run
//! # async fn example() -> anyhow::Result<()> {
//! # let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
//! let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await?;
//!
//! // Check if BC compression is supported
//! let features = adapter.features();
//! if !features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
//!     panic!("BC texture compression not supported");
//! }
//!
//! let (device, queue) = adapter.request_device(
//!     &wgpu::DeviceDescriptor {
//!         label: None,
//!         required_features: wgpu::Features::TEXTURE_COMPRESSION_BC,
//!         required_limits: wgpu::Limits::default(),
//!         memory_hints: wgpu::MemoryHints::default(),
//!         trace: wgpu::Trace::Off,
//!         experimental_features: wgpu::ExperimentalFeatures::default(),
//!     },
//! ).await?;
//! # Ok(())
//! # }
//! ```

// Re-export TextureFormat and HAP format types for convenience
pub use hap_parser::TextureFormat;
pub use hap_qt::{CompressionMode, DxtQuality, HapFormat, HapFrameEncoder, QtHapReader, VideoConfig};
use parking_lot::{Condvar, Mutex};
use std::borrow::Cow;
use std::collections::VecDeque;
use std::sync::Arc;
use std::thread;
use std::time::Instant;
use thiserror::Error;

// Encoder modules
pub mod encoder;
pub mod gpu_compress;
pub use encoder::{EncodeConfig, HapEncoderBuilder, HapVideoEncoder, VideoEncoderError, pad_rgba};
pub use gpu_compress::{GpuDxtCompressor, GpuCompressError};

/// WGSL helper for displaying Hap Q frames.
///
/// Hap Q stores scaled YCoCg in a BC3 texture (`r=Co, g=Cg, b=scale, a=Y`),
/// so a sampled texel is not RGB — every format where
/// [`TextureFormat::needs_ycocg_convert`] is true must run this conversion in
/// the fragment shader before display. Paste this snippet into your shader and
/// call `hap_ycocg_to_rgb(textureSample(...))`. This is the canonical HAP
/// scaled-YCoCg decode; it matches files from ffmpeg, Resolume and VDMX, and is
/// the exact inverse of this workspace's HapY encoder.
pub const YCOCG_TO_RGB_WGSL: &str = r#"
fn hap_ycocg_to_rgb(c: vec4<f32>) -> vec3<f32> {
    let offset = 128.0 / 255.0;
    let scale = (c.b * (255.0 / 8.0)) + 1.0;
    let co = (c.r - offset) / scale;
    let cg = (c.g - offset) / scale;
    let y = c.a;
    return vec3<f32>(y + co - cg, y + cg, y - co - cg);
}
"#;

/// Errors that can occur during HAP playback
#[derive(Error, Debug)]
pub enum HapPlayerError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("HAP parse error: {0}")]
    HapError(#[from] hap_parser::HapError),
    
    #[error("QuickTime error: {0}")]
    QtError(#[from] hap_qt::QtError),
    
    #[error("Invalid state: {0}")]
    InvalidState(String),
    
    #[error("BC texture compression not supported. HAP requires TEXTURE_COMPRESSION_BC feature.")]
    BcCompressionNotSupported,
}

/// Playback state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaybackState {
    Stopped,
    Playing,
    Paused,
}

/// Loop mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopMode {
    None,
    Loop,
    Palindrome,
}

/// A GPU texture containing a HAP frame
pub struct HapTexture {
    /// The wgpu texture (Arc for sharing)
    pub texture: Arc<wgpu::Texture>,
    /// Texture view (Arc for sharing)
    pub view: Arc<wgpu::TextureView>,
    /// Frame index
    pub frame_index: u32,
    /// Texture format (for color space conversion)
    pub format: TextureFormat,
}

impl HapTexture {
    /// Create a `HapTexture` from raw decompressed BCn block data.
    ///
    /// `data` should already match `width`×`height` for `format` — use
    /// [`hap_parser::detect_format`] so the format is correct. If the size does
    /// not match, the buffer is padded or trimmed to the expected size and a
    /// warning is logged, so a corrupt frame degrades gracefully instead of
    /// panicking or overrunning the GPU copy.
    pub fn from_dxt_data(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        format: TextureFormat,
        data: &[u8],
        frame_index: u32,
    ) -> Self {
        let wgpu_format = hap_format_to_wgpu(format);
        let blocks_x = width.div_ceil(4);
        let blocks_y = height.div_ceil(4);
        let bytes_per_block = format.bytes_per_block();
        let expected_size = (blocks_x * blocks_y) as usize * bytes_per_block;

        // With correct format detection upstream the data is already the right
        // size. A remaining mismatch means a corrupt/truncated frame.
        // ponytail: replaces a ~180-line dimension-guessing search — if silent
        // pad/trim ever proves wrong, surface a decode error to the caller instead.
        let data: Cow<[u8]> = if data.len() == expected_size {
            Cow::Borrowed(data)
        } else {
            eprintln!(
                "hap-wgpu: frame {frame_index} size mismatch ({width}x{height} {format:?}): \
                 got {} bytes, expected {expected_size} — padding/trimming",
                data.len()
            );
            let mut buf = vec![0u8; expected_size];
            let n = data.len().min(expected_size);
            buf[..n].copy_from_slice(&data[..n]);
            Cow::Owned(buf)
        };

        let bytes_per_row = blocks_x * bytes_per_block as u32;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("HAP Frame {frame_index}")),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(blocks_y),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            texture: Arc::new(texture),
            view: Arc::new(view),
            frame_index,
            format,
        }
    }
}

/// Convert HAP texture format to wgpu format
fn hap_format_to_wgpu(format: TextureFormat) -> wgpu::TextureFormat {
    match format {
        TextureFormat::RgbDxt1 => wgpu::TextureFormat::Bc1RgbaUnorm,
        TextureFormat::RgbaDxt5 => wgpu::TextureFormat::Bc3RgbaUnorm,
        TextureFormat::YcoCgDxt5 => wgpu::TextureFormat::Bc3RgbaUnorm,
        TextureFormat::AlphaRgtc1 => wgpu::TextureFormat::Bc4RUnorm,
        TextureFormat::RgbaBc7 => wgpu::TextureFormat::Bc7RgbaUnorm,
    }
}

/// Calculate dimensions padded to multiple of 4 (required for DXT)
pub fn padded_dimensions(width: u32, height: u32) -> (u32, u32) {
    let padded_width = width.div_ceil(4) * 4;
    let padded_height = height.div_ceil(4) * 4;
    (padded_width, padded_height)
}

/// Returns the wgpu features required for HAP playback
pub fn required_features() -> wgpu::Features {
    wgpu::Features::TEXTURE_COMPRESSION_BC
}

/// Check if the adapter supports HAP playback
pub fn is_supported(adapter: &wgpu::Adapter) -> bool {
    adapter.features().contains(wgpu::Features::TEXTURE_COMPRESSION_BC)
}

/// Get a human-readable error message if HAP is not supported
pub fn check_support(adapter: &wgpu::Adapter) -> Result<(), String> {
    let features = adapter.features();
    
    if !features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
        return Err(
            "BC texture compression not supported by this GPU. \
             HAP playback requires TEXTURE_COMPRESSION_BC feature."
            .to_string()
        );
    }
    
    Ok(())
}

/// Shared state between player and decoder thread
struct PlayerState {
    /// Current playback position
    current_frame: f32,
    /// Playback speed (1.0 = normal, -1.0 = reverse)
    speed: f32,
    /// Playback state
    state: PlaybackState,
    /// Loop mode
    loop_mode: LoopMode,
    /// Frame cache (keeps recent frames)
    frame_cache: VecDeque<(u32, Arc<HapTexture>)>,
    /// Decoded but not yet uploaded, in frame order. The decoder thread fills
    /// this; `update()` drains it. The upload deliberately stays on the
    /// caller's thread — doing it here from a second thread costs more in
    /// queue contention and staging memory than the decode it saves.
    decoded: VecDeque<(u32, hap_qt::HapFrame)>,
    /// Maximum cache size
    max_cache_size: usize,
    /// Last frame time
    last_frame_time: Instant,
    /// The last texture handed out, shown again if the decoder has not caught
    /// up. Repeating a frame beats blocking the render thread for one.
    last_texture: Option<Arc<HapTexture>>,
    /// Which way the prefetcher should read. Follows playback, so reverse
    /// playback fills backwards.
    forward: bool,
    /// The frame `update()` has taken out of `decoded` and is uploading with
    /// the lock released. It is in neither queue meanwhile; without this the
    /// decoder decodes it again — a third more decodes on a 4K clip, where the
    /// upload is an 8 MB copy and the window is wide.
    uploading: Option<u32>,
    /// The playhead frame the decoder was last woken for; `update()` wakes it
    /// again only when the playhead reaches a new frame.
    woken_for: Option<u32>,
    /// How far the playhead moves per `update()`, smoothed. Read-ahead steps
    /// by this, so a caller slower than the clip (a 30 Hz loop on a 50 fps
    /// clip, a loaded render thread) only decodes the frames it will show.
    stride: f32,
}

impl PlayerState {
    fn new() -> Self {
        Self {
            current_frame: 0.0,
            speed: 1.0,
            state: PlaybackState::Stopped,
            loop_mode: LoopMode::None,
            frame_cache: VecDeque::new(),
            decoded: VecDeque::new(),
            max_cache_size: 8,
            last_frame_time: Instant::now(),
            last_texture: None,
            forward: true,
            uploading: None,
            woken_for: None,
            stride: 1.0,
        }
    }
}

/// How many frames ahead of the playhead the decoder keeps ready. Two is
/// enough to cover a decode that takes longer than a frame without holding a
/// large clip's worth of GPU textures: a 4K HapY frame is ~8 MB.
const PREFETCH: u32 = 3;

/// Frames the decode thread has actually decompressed, for
/// [`HapPlayer::decode_count`]. Every decode is a Snappy pass over a whole
/// frame, so this is the number that decides CPU cost — and the one to watch
/// when changing how far ahead the thread reads.
#[derive(Default)]
pub(crate) struct DecodeCounter(std::sync::atomic::AtomicU64);

impl DecodeCounter {
    fn bump(&self) {
        self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    fn get(&self) -> u64 {
        self.0.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// The frames the caller will ask for on its next `PREFETCH + 1` calls: where
/// the playhead *will be*, not simply the next frames. At a stride of 1.67 (a
/// 30 Hz caller on a 50 fps clip) that is 0, 1, 3, 5, not 0, 1, 2, 3 — the
/// frames between are never shown, so decoding them is waste. Looping playback
/// wraps; otherwise the ends clamp.
fn predicted_frames(
    pos: f32,
    stride: f32,
    forward: bool,
    wraps: bool,
    frame_count: u32,
) -> impl Iterator<Item = u32> {
    let n = i64::from(frame_count.max(1));
    let sign = if forward { 1.0 } else { -1.0 };
    (0..=PREFETCH).map(move |step| {
        let f = (pos + sign * step as f32 * stride.max(1.0)).floor() as i64;
        (if wraps { f.rem_euclid(n) } else { f.clamp(0, n - 1) }) as u32
    })
}

/// Keep the cache filled ahead of the playhead until `running` clears.
///
/// Reads happen here, on this thread, so the caller's `update()` is a lookup
/// plus an upload. The playhead the caller advances *is* the request: when
/// nothing is wanted the thread sleeps on `wake`, which `update()` signals when
/// the playhead reaches a new frame and every control signals when it changes
/// what is wanted. (It used to poll every 2 ms; that alone cost 1.5-4 points
/// of CPU over decoding inline.)
fn decode_ahead(
    mut reader: QtHapReader,
    state: Arc<Mutex<PlayerState>>,
    wake: Arc<Condvar>,
    running: Arc<std::sync::atomic::AtomicBool>,
    frame_count: u32,
    decodes: Arc<DecodeCounter>,
) {
    use std::sync::atomic::Ordering;
    while running.load(Ordering::Relaxed) {
        // What is wanted, and what is already decoded. The lock is held for
        // the look, never for the read.
        let wanted = {
            let mut state = state.lock();
            let pos = state.current_frame;
            let target = pos as u32;
            let sign = if state.forward { 1.0 } else { -1.0 };
            let stride = state.stride.max(1.0);
            // Looping playback wraps, so read ahead across the loop point too;
            // otherwise every loop starts on a miss.
            let wraps = state.loop_mode == LoopMode::Loop;
            let n = i64::from(frame_count.max(1));
            // Drop what the playhead has passed, or a clip longer than the
            // queue would evict the frames just fetched.
            let span = (PREFETCH as f32 * stride).ceil() as i64;
            state.decoded.retain(|(idx, _)| {
                let d = (i64::from(*idx) - i64::from(target)) * sign as i64;
                let ahead = if wraps { d.rem_euclid(n) } else { d };
                (0..=span).contains(&ahead)
            });
            let found = predicted_frames(pos, stride, sign > 0.0, wraps, frame_count)
                // Skip anything already decoded *or* already uploaded.
                // `update()` takes a frame out of `decoded` to upload it, but
                // the playhead has not moved on yet — without the cache check
                // the very next poll decodes that same frame again, doubling
                // the work for a frame that is already on the GPU.
                .find(|f| {
                    state.uploading != Some(*f)
                        && !state.decoded.iter().any(|(idx, _)| idx == f)
                        && !state.frame_cache.iter().any(|(idx, _)| idx == f)
                });
            if found.is_none() {
                // Full: sleep until the playhead moves or a control changes what
                // is wanted. Checked and waited under one lock, so no wakeup is
                // lost; the timeout only bounds how long shutdown waits.
                wake.wait_for(&mut state, std::time::Duration::from_millis(100));
            }
            found
        };

        let Some(frame) = wanted else {
            continue;
        };

        decodes.bump();
        match reader.read_frame(frame) {
            Ok(hap_frame) => {
                let mut state = state.lock();
                if !state.decoded.iter().any(|(idx, _)| *idx == frame) {
                    state.decoded.push_back((frame, hap_frame));
                    while state.decoded.len() > PREFETCH as usize + 1 {
                        state.decoded.pop_front();
                    }
                }
            }
            Err(e) => {
                eprintln!("Failed to decode frame {}: {}", frame, e);
                // A bad frame must not spin this loop at full tilt.
                thread::sleep(std::time::Duration::from_millis(20));
            }
        }
    }
}

/// HAP Video Player with background loading
pub struct HapPlayer {
    /// Texture format, read from the reader before it moved to the decoder.
    texture_format: TextureFormat,
    /// wgpu device
    device: Arc<wgpu::Device>,
    /// wgpu queue
    queue: Arc<wgpu::Queue>,
    /// Video dimensions
    dimensions: (u32, u32),
    /// Padded dimensions (multiple of 4)
    padded_dimensions: (u32, u32),
    /// Frame count
    frame_count: u32,
    /// Frame rate
    fps: f32,
    /// Duration
    duration: f64,
    /// Codec type
    codec_type: String,
    /// Shared state
    state: Arc<Mutex<PlayerState>>,
    /// Wakes the decoder thread when what it should read changes.
    wake: Arc<Condvar>,
    /// Cleared on drop to stop the decoder thread.
    running: Arc<std::sync::atomic::AtomicBool>,
    /// Decoder thread handle — reads and uploads ahead of the playhead so
    /// `update()` never decodes on the caller's thread.
    decoder_thread: Option<thread::JoinHandle<()>>,
    /// How many frames the decoder has decompressed. See [`Self::decode_count`].
    decode_counter: Arc<DecodeCounter>,
}

impl HapPlayer {
    /// Open a HAP video file and create a player
    ///
    /// # Panics
    /// 
    /// This function will panic if the wgpu device does not support
    /// `TEXTURE_COMPRESSION_BC` feature. Use `check_support()` or 
    /// `is_supported()` to verify before calling this function.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example() -> anyhow::Result<()> {
    /// # let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    /// // First, check adapter support when creating device
    /// let adapter = instance.request_adapter(&Default::default()).await.unwrap();
    /// 
    /// if !hap_wgpu::is_supported(&adapter) {
    ///     panic!("HAP not supported on this GPU");
    /// }
    ///
    /// let (device, queue) = adapter.request_device(
    ///     &wgpu::DeviceDescriptor {
    ///         label: None,
    ///         required_features: hap_wgpu::required_features(),
    ///         required_limits: wgpu::Limits::default(),
    ///         memory_hints: wgpu::MemoryHints::default(),
    ///         trace: wgpu::Trace::Off,
    ///         experimental_features: wgpu::ExperimentalFeatures::default(),
    ///     },
    /// ).await.unwrap();
    ///
    /// // Now safe to open HAP files
    /// let player = hap_wgpu::HapPlayer::open("video.mov", std::sync::Arc::new(device), std::sync::Arc::new(queue))?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn open<P: AsRef<std::path::Path>>(
        path: P,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    ) -> Result<Self, HapPlayerError> {
        // Note: We can't directly check device features, but we document
        // that callers must check adapter support before creating the device
        let reader = QtHapReader::open(path)?;
        
        let dimensions = reader.resolution();
        let padded_dimensions = padded_dimensions(dimensions.0, dimensions.1);
        let frame_count = reader.frame_count();
        let fps = reader.fps();
        let duration = reader.duration();
        let codec_type = reader.codec_type().to_string();
        
        println!(
            "HapPlayer: {}x{} (padded to {}x{}), {} frames @ {}fps, codec: {}",
            dimensions.0, dimensions.1,
            padded_dimensions.0, padded_dimensions.1,
            frame_count, fps, codec_type
        );
        
        let state = Arc::new(Mutex::new(PlayerState::new()));
        let texture_format = reader.texture_format();
        let running = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let decode_counter = Arc::new(DecodeCounter::default());

        // The reader moves to the decoder thread: every read and upload happens
        // there, ahead of the playhead, so `update()` is a cache lookup. Doing
        // it inline was ~80% of the host's render-thread work on a 4K clip.
        let wake = Arc::new(Condvar::new());
        let decoder_thread = Some(thread::spawn({
            let state = Arc::clone(&state);
            let wake = Arc::clone(&wake);
            let running = Arc::clone(&running);
            let decodes = Arc::clone(&decode_counter);
            move || decode_ahead(reader, state, wake, running, frame_count, decodes)
        }));

        Ok(Self {
            decode_counter,
            texture_format,
            device,
            queue,
            dimensions,
            padded_dimensions,
            frame_count,
            fps,
            duration,
            codec_type,
            state,
            wake,
            running,
            decoder_thread,
        })
    }
    
    /// Start playback
    pub fn play(&mut self) {
        let mut state = self.state.lock();
        state.state = PlaybackState::Playing;
        state.last_frame_time = Instant::now();
        drop(state);
        self.wake.notify_one();
    }
    
    /// Pause playback
    pub fn pause(&mut self) {
        let mut state = self.state.lock();
        state.state = PlaybackState::Paused;
        drop(state);
        self.wake.notify_one();
    }
    
    /// Stop playback
    pub fn stop(&mut self) {
        let mut state = self.state.lock();
        state.state = PlaybackState::Stopped;
        state.current_frame = 0.0;
        state.frame_cache.clear();
        drop(state);
        self.wake.notify_one();
    }
    
    /// Set playback speed
    pub fn set_speed(&mut self, speed: f32) {
        let mut state = self.state.lock();
        state.speed = speed;
        drop(state);
        self.wake.notify_one();
    }
    
    /// Set loop mode
    pub fn set_loop_mode(&mut self, mode: LoopMode) {
        let mut state = self.state.lock();
        state.loop_mode = mode;
        drop(state);
        self.wake.notify_one();
    }
    
    /// Frames decompressed since the player opened.
    ///
    /// One Snappy pass over a whole frame each, so this is what decides CPU
    /// cost. In steady state it should track the clip's frame rate; materially
    /// above that means the read-ahead is fetching frames nobody displays.
    pub fn decode_count(&self) -> u64 {
        self.decode_counter.get()
    }

    /// Seek to specific frame
    pub fn seek_to_frame(&mut self, frame: u32) {
        let mut state = self.state.lock();
        state.current_frame = frame.min(self.frame_count - 1) as f32;
        state.frame_cache.clear();
        drop(state);
        self.wake.notify_one();
    }
    
    /// Get current frame texture (call every render frame)
    pub fn update(&mut self) -> Option<Arc<HapTexture>> {
        let mut state = self.state.lock();
        
        // Update playback position
        if state.state == PlaybackState::Playing {
            let now = Instant::now();
            let dt = now - state.last_frame_time;
            state.last_frame_time = now;
            
            // Advance frame
            let frame_delta = state.speed * self.fps * dt.as_secs_f32();
            state.current_frame += frame_delta;
            state.stride = 0.8 * state.stride + 0.2 * frame_delta.abs();
            
            // Handle loop/end
            if state.current_frame >= self.frame_count as f32 {
                match state.loop_mode {
                    LoopMode::None => {
                        state.current_frame = (self.frame_count - 1) as f32;
                        state.state = PlaybackState::Stopped;
                    }
                    LoopMode::Loop => {
                        state.current_frame -= self.frame_count as f32;
                    }
                    LoopMode::Palindrome => {
                        state.current_frame = self.frame_count as f32 - 1.0;
                        state.speed = -state.speed.abs();
                    }
                }
            } else if state.current_frame < 0.0 {
                match state.loop_mode {
                    LoopMode::None => {
                        state.current_frame = 0.0;
                        state.state = PlaybackState::Stopped;
                    }
                    LoopMode::Loop => {
                        state.current_frame += self.frame_count as f32;
                    }
                    LoopMode::Palindrome => {
                        state.current_frame = 0.0;
                        state.speed = state.speed.abs();
                    }
                }
            }
        }
        
        let target_frame = state.current_frame as u32;
        state.forward = state.speed >= 0.0;
        if state.woken_for != Some(target_frame) {
            state.woken_for = Some(target_frame);
            self.wake.notify_one();
        }

        // A cache lookup, and nothing else: the decoder thread is what fills
        // it. A miss shows the previous frame rather than stalling the caller
        // for a read and a decompress — on a 4K clip that is milliseconds, and
        // the caller is usually a render loop with a frame to finish.
        let hit = state
            .frame_cache
            .iter()
            .find(|(idx, _)| *idx == target_frame)
            .map(|(_, texture)| Arc::clone(texture));
        if let Some(texture) = hit {
            state.last_texture = Some(Arc::clone(&texture));
            return Some(texture);
        }

        // Not uploaded yet: take the decoded bytes the thread left and upload
        // them here, on the caller's thread, where the queue already is.
        let Some(pos) = state.decoded.iter().position(|(idx, _)| *idx == target_frame) else {
            // The decoder has not caught up. Showing the previous frame beats
            // blocking a render loop for a read and a decompress.
            return state.last_texture.clone();
        };
        let (frame, hap_frame) = state.decoded.remove(pos).expect("position just found");
        state.uploading = Some(frame);
        drop(state);

        let texture = Arc::new(HapTexture::from_dxt_data(
            &self.device,
            &self.queue,
            self.padded_dimensions.0,
            self.padded_dimensions.1,
            hap_frame.format,
            &hap_frame.data,
            frame,
        ));
        let mut state = self.state.lock();
        state.uploading = None;
        state.frame_cache.push_back((frame, Arc::clone(&texture)));
        while state.frame_cache.len() > state.max_cache_size {
            state.frame_cache.pop_front();
        }
        state.last_texture = Some(Arc::clone(&texture));
        Some(texture)
    }
    
    /// Get video dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }
    
    /// Get padded dimensions (multiple of 4)
    pub fn padded_dimensions(&self) -> (u32, u32) {
        self.padded_dimensions
    }
    
    /// Get frame count
    pub fn frame_count(&self) -> u32 {
        self.frame_count
    }
    
    /// Get frame rate
    pub fn fps(&self) -> f32 {
        self.fps
    }
    
    /// Get duration
    pub fn duration(&self) -> f64 {
        self.duration
    }
    
    /// Get codec type
    pub fn codec_type(&self) -> &str {
        &self.codec_type
    }
    
    /// Get texture format
    pub fn texture_format(&self) -> TextureFormat {
        self.texture_format
    }
    
    /// Get current playback state
    pub fn playback_state(&self) -> PlaybackState {
        self.state.lock().state
    }
    
    /// Check if currently playing
    pub fn is_playing(&self) -> bool {
        self.state.lock().state == PlaybackState::Playing
    }
}

impl Drop for HapPlayer {
    fn drop(&mut self) {
        self.running
            .store(false, std::sync::atomic::Ordering::Relaxed);
        self.wake.notify_one();
        if let Some(thread) = self.decoder_thread.take() {
            // At most one poll interval plus a frame decode, so bounded.
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_ahead_follows_the_playhead_stride() {
        let frames = |pos, stride, fwd, wraps| {
            predicted_frames(pos, stride, fwd, wraps, 10).collect::<Vec<_>>()
        };
        // A caller at the clip rate wants consecutive frames.
        assert_eq!(frames(0.0, 1.0, true, false), [0, 1, 2, 3]);
        // A caller slower than the clip skips the frames it will never show.
        assert_eq!(frames(0.0, 5.0 / 3.0, true, false), [0, 1, 3, 5]);
        // A caller faster than the clip still reads whole frames ahead.
        assert_eq!(frames(4.2, 0.5, true, false), [4, 5, 6, 7]);
        // Looping wraps past the end; one-shot clamps at it.
        assert_eq!(frames(8.0, 1.0, true, true), [8, 9, 0, 1]);
        assert_eq!(frames(8.0, 1.0, true, false), [8, 9, 9, 9]);
        // Reverse reads backwards, wrapping or clamping at frame 0.
        assert_eq!(frames(1.0, 1.0, false, true), [1, 0, 9, 8]);
        assert_eq!(frames(1.0, 1.0, false, false), [1, 0, 0, 0]);
    }

    #[test]
    fn test_padded_dimensions() {
        assert_eq!(padded_dimensions(4, 4), (4, 4));
        assert_eq!(padded_dimensions(5, 5), (8, 8));
        assert_eq!(padded_dimensions(1920, 1080), (1920, 1080));
        assert_eq!(padded_dimensions(1920, 1081), (1920, 1084));
    }

    #[test]
    fn test_hap_format_to_wgpu() {
        assert_eq!(
            hap_format_to_wgpu(TextureFormat::RgbDxt1),
            wgpu::TextureFormat::Bc1RgbaUnorm
        );
        assert_eq!(
            hap_format_to_wgpu(TextureFormat::RgbaDxt5),
            wgpu::TextureFormat::Bc3RgbaUnorm
        );
    }
}
