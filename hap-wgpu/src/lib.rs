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
use parking_lot::Mutex;
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
    /// Maximum cache size
    max_cache_size: usize,
    /// Last frame time
    last_frame_time: Instant,
}

impl PlayerState {
    fn new() -> Self {
        Self {
            current_frame: 0.0,
            speed: 1.0,
            state: PlaybackState::Stopped,
            loop_mode: LoopMode::None,
            frame_cache: VecDeque::new(),
            max_cache_size: 8,
            last_frame_time: Instant::now(),
        }
    }
}

/// HAP Video Player with background loading
pub struct HapPlayer {
    /// Video reader
    reader: QtHapReader,
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
    /// Decoder thread handle
    _decoder_thread: Option<thread::JoinHandle<()>>,
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
        
        Ok(Self {
            reader,
            device,
            queue,
            dimensions,
            padded_dimensions,
            frame_count,
            fps,
            duration,
            codec_type,
            state,
            _decoder_thread: None,
        })
    }
    
    /// Start playback
    pub fn play(&mut self) {
        let mut state = self.state.lock();
        state.state = PlaybackState::Playing;
        state.last_frame_time = Instant::now();
    }
    
    /// Pause playback
    pub fn pause(&mut self) {
        let mut state = self.state.lock();
        state.state = PlaybackState::Paused;
    }
    
    /// Stop playback
    pub fn stop(&mut self) {
        let mut state = self.state.lock();
        state.state = PlaybackState::Stopped;
        state.current_frame = 0.0;
        state.frame_cache.clear();
    }
    
    /// Set playback speed
    pub fn set_speed(&mut self, speed: f32) {
        let mut state = self.state.lock();
        state.speed = speed;
    }
    
    /// Set loop mode
    pub fn set_loop_mode(&mut self, mode: LoopMode) {
        let mut state = self.state.lock();
        state.loop_mode = mode;
    }
    
    /// Seek to specific frame
    pub fn seek_to_frame(&mut self, frame: u32) {
        let mut state = self.state.lock();
        state.current_frame = frame.min(self.frame_count - 1) as f32;
        state.frame_cache.clear();
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
        
        // Check cache first
        for (frame_idx, texture) in &state.frame_cache {
            if *frame_idx == target_frame {
                return Some(texture.clone());
            }
        }
        
        // Not in cache, decode and upload
        drop(state); // Release lock during decode
        
        match self.decode_and_upload(target_frame) {
            Some(texture) => {
                let mut state = self.state.lock();
                
                // Add to cache
                state.frame_cache.push_back((target_frame, texture.clone()));
                
                // Trim cache if needed
                while state.frame_cache.len() > state.max_cache_size {
                    state.frame_cache.pop_front();
                }
                
                Some(texture)
            }
            None => None,
        }
    }
    
    /// Decode a frame and upload to GPU
    fn decode_and_upload(&mut self, frame: u32) -> Option<Arc<HapTexture>> {
        match self.reader.read_frame(frame) {
            Ok(hap_frame) => {
                let texture = HapTexture::from_dxt_data(
                    &self.device,
                    &self.queue,
                    self.padded_dimensions.0,
                    self.padded_dimensions.1,
                    hap_frame.format,
                    &hap_frame.data,
                    frame,
                );
                Some(Arc::new(texture))
            }
            Err(e) => {
                eprintln!("Failed to decode frame {}: {}", frame, e);
                None
            }
        }
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
        self.reader.texture_format()
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

#[cfg(test)]
mod tests {
    use super::*;

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
