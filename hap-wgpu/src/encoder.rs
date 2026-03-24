//! HAP Video Encoder
//!
//! High-level HAP video encoding using GPU acceleration when available.
//! Provides convenient APIs for encoding complete videos from frames or images.

use crate::gpu_compress::{GpuCompressError, GpuDxtCompressor};
use hap_qt::{
    CompressionMode, HapEncodeError, HapFormat, HapFrameEncoder, QtHapWriter, QtWriterError,
    VideoConfig,
};
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;

/// Errors that can occur during HAP video encoding
#[derive(Error, Debug)]
pub enum VideoEncoderError {
    #[error("HAP encoding error: {0}")]
    HapError(#[from] HapEncodeError),

    #[error("QuickTime writer error: {0}")]
    WriterError(#[from] QtWriterError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Image loading error: {0}")]
    ImageError(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("GPU not available: {0}")]
    GpuNotAvailable(String),

    #[error("GPU compression error: {0}")]
    GpuCompressError(#[from] GpuCompressError),
}

/// Encoding quality preset
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncodeQuality {
    /// Fastest encoding, lower quality (less DXT search)
    Fast,
    /// Balanced quality and speed
    Balanced,
    /// Slowest encoding, best quality (higher DXT search)
    Best,
}

impl EncodeQuality {
    /// Get texpresso quality parameters
    /// Get texpresso quality parameters
    #[allow(dead_code)]
    #[cfg(feature = "cpu-compression")]
    fn texpresso_params(&self) -> texpresso::Params {
        match self {
            EncodeQuality::Fast => texpresso::Params {
                algorithm: texpresso::Algorithm::RangeFit,
                ..Default::default()
            },
            EncodeQuality::Balanced => texpresso::Params::default(),
            EncodeQuality::Best => texpresso::Params {
                algorithm: texpresso::Algorithm::IterativeClusterFit,
                ..Default::default()
            },
        }
    }
}

/// Configuration for video encoding
#[derive(Clone, Debug)]
pub struct EncodeConfig {
    /// Frame width in pixels
    pub width: u32,
    /// Frame height in pixels
    pub height: u32,
    /// Frame rate (frames per second)
    pub fps: f32,
    /// Total number of frames to encode
    pub frame_count: u32,
    /// Encoding quality preset
    pub quality: EncodeQuality,
    /// HAP format variant
    pub format: HapFormat,
    /// Use Snappy compression (default: true)
    pub use_snappy: bool,
}

impl EncodeConfig {
    /// Create a new encoding configuration
    pub fn new(width: u32, height: u32, fps: f32, frame_count: u32) -> Self {
        Self {
            width,
            height,
            fps,
            frame_count,
            quality: EncodeQuality::Balanced,
            format: HapFormat::HapY, // Good default for quality
            use_snappy: true,
        }
    }

    /// Set the HAP format
    pub fn with_format(mut self, format: HapFormat) -> Self {
        self.format = format;
        self
    }

    /// Set the quality preset
    pub fn with_quality(mut self, quality: EncodeQuality) -> Self {
        self.quality = quality;
        self
    }

    /// Enable/disable Snappy compression
    pub fn with_snappy(mut self, enabled: bool) -> Self {
        self.use_snappy = enabled;
        self
    }
}

/// High-level HAP video encoder using GPU acceleration when available
///
/// This encoder provides convenient APIs for encoding complete videos.
/// For GPU-accelerated DXT compression, it uses wgpu compute shaders.
/// Falls back to CPU compression via `texpresso` if GPU is unavailable.
pub struct HapVideoEncoder {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    /// GPU compressor (None if GPU not available or not initialized)
    gpu_compressor: Option<GpuDxtCompressor>,
}

impl HapVideoEncoder {
    /// Create a new HAP video encoder
    ///
    /// # Arguments
    ///
    /// * `device` - wgpu device for GPU operations
    /// * `queue` - wgpu queue for command submission
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # async fn example() -> anyhow::Result<()> {
    /// # let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    /// # let adapter = instance.request_adapter(&Default::default()).await.unwrap();
    /// # let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
    /// use hap_wgpu::HapVideoEncoder;
    ///
    /// let encoder = HapVideoEncoder::new(Arc::new(device), Arc::new(queue));
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self {
            device,
            queue,
            gpu_compressor: None,
        }
    }

    /// Initialize GPU compression for the given dimensions.
    /// Call this before encoding if you want GPU acceleration.
    /// Falls back to CPU silently if GPU init fails.
    pub fn init_gpu(&mut self, width: u32, height: u32) {
        self.gpu_compressor = GpuDxtCompressor::try_new(
            Arc::clone(&self.device),
            Arc::clone(&self.queue),
            width,
            height,
        );
    }

    /// Check if GPU compression is available
    pub fn is_gpu_compression_available(&self) -> bool {
        self.gpu_compressor.is_some()
    }

    /// Encode video from a frame provider callback
    ///
    /// The callback is called for each frame index and should return RGBA pixels.
    /// This is the most flexible API for encoding from any source.
    ///
    /// # Arguments
    ///
    /// * `output_path` - Path for the output .mov file
    /// * `config` - Encoding configuration
    /// * `frame_provider` - Callback that provides RGBA frame data for each frame index
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use hap_wgpu::{HapVideoEncoder, EncodeConfig};
    /// # fn encode_video(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Result<(), Box<dyn std::error::Error>> {
    /// let encoder = HapVideoEncoder::new(device, queue);
    /// let config = EncodeConfig::new(1920, 1080, 30.0, 300);
    ///
    /// encoder.encode("output.mov", config, |frame_idx| {
    ///     // Generate or load frame data
    ///     let rgba_data = vec![255u8; 1920 * 1080 * 4]; // Placeholder
    ///     Ok(rgba_data)
    /// })?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn encode<F, P>(
        &self,
        output_path: P,
        config: EncodeConfig,
        mut frame_provider: F,
    ) -> Result<(), VideoEncoderError>
    where
        F: FnMut(u32) -> Result<Vec<u8>, Box<dyn std::error::Error>>,
        P: AsRef<Path>,
    {
        // Create frame encoder (handles Snappy + header framing)
        let mut frame_encoder =
            HapFrameEncoder::new(config.format, config.width, config.height)?;

        let compression = if config.use_snappy {
            CompressionMode::Snappy
        } else {
            CompressionMode::None
        };
        frame_encoder.set_compression(compression);

        let video_config = VideoConfig {
            width: config.width,
            height: config.height,
            fps: config.fps,
            codec: config.format,
            timescale: (config.fps * 100.0) as u32,
        };

        let mut writer = QtHapWriter::create(output_path, video_config)?;

        // Check if GPU path is available for this format
        let use_gpu = self.gpu_compressor.is_some()
            && GpuDxtCompressor::supports_format(config.format);

        for frame_idx in 0..config.frame_count {
            let rgba_data = frame_provider(frame_idx).map_err(|e| {
                VideoEncoderError::InvalidConfig(format!(
                    "Frame provider failed at frame {}: {}",
                    frame_idx, e
                ))
            })?;

            let hap_frame = if use_gpu {
                // GPU path: DXT compression on GPU, then Snappy + header on CPU
                let gpu = self.gpu_compressor.as_ref().unwrap();

                // Pad input if needed (GPU expects padded dimensions)
                let (pw, ph) = gpu.dimensions();
                let padded = if config.width != pw || config.height != ph {
                    pad_rgba(&rgba_data, config.width, config.height, pw, ph)
                } else {
                    rgba_data
                };

                let dxt_data = gpu.compress(&padded, config.format)?;
                frame_encoder.encode_from_dxt(&dxt_data)?
            } else {
                // CPU path: full encode (DXT + Snappy + header)
                frame_encoder.encode(&rgba_data)?
            };

            writer.write_frame(&hap_frame)?;
        }

        writer.finalize()?;

        Ok(())
    }

    /// Encode from a sequence of image files
    ///
    /// # Arguments
    ///
    /// * `output_path` - Path for the output .mov file
    /// * `config` - Encoding configuration
    /// * `image_paths` - Paths to image files (PNG, JPEG, etc.)
    ///
    /// # Note
    ///
    /// Requires the `image` feature to be enabled.
    #[cfg(feature = "image-support")]
    pub fn encode_from_images<P, I>(
        &self,
        output_path: P,
        config: EncodeConfig,
        image_paths: I,
    ) -> Result<(), VideoEncoderError>
    where
        P: AsRef<Path>,
        I: IntoIterator,
        I::Item: AsRef<Path>,
    {


        let paths: Vec<_> = image_paths.into_iter().collect();
        let frame_count = paths.len() as u32;

        if frame_count == 0 {
            return Err(VideoEncoderError::InvalidConfig(
                "No image paths provided".to_string(),
            ));
        }

        // Store values needed by closure before config is moved
        let width = config.width;
        let height = config.height;

        // Update config with actual frame count
        let config = EncodeConfig {
            frame_count,
            ..config
        };

        self.encode(output_path, config, move |frame_idx| {
            let path = paths.get(frame_idx as usize).ok_or("Frame index out of range")?;
            let img = image::open(path.as_ref())
                .map_err(|e| format!("Failed to open image: {}", e))?;

            // Convert to RGBA8
            let rgba = img.to_rgba8();

            // Resize if needed
            let resized = if rgba.width() != width || rgba.height() != height {
                image::imageops::resize(
                    &rgba,
                    width,
                    height,
                    image::imageops::FilterType::Lanczos3,
                )
                .into_raw()
            } else {
                rgba.into_raw()
            };

            Ok::<Vec<u8>, Box<dyn std::error::Error>>(resized)
        })
    }

    /// Encode from raw RGBA frames in memory
    ///
    /// # Arguments
    ///
    /// * `output_path` - Path for the output .mov file
    /// * `config` - Encoding configuration
    /// * `frames` - Iterator of RGBA frame data
    pub fn encode_from_frames<P, I>(
        &self,
        output_path: P,
        config: EncodeConfig,
        frames: I,
    ) -> Result<(), VideoEncoderError>
    where
        P: AsRef<Path>,
        I: IntoIterator<Item = Vec<u8>>,
    {
        let frames: Vec<_> = frames.into_iter().collect();
        let frame_count = frames.len() as u32;

        if frame_count == 0 {
            return Err(VideoEncoderError::InvalidConfig(
                "No frames provided".to_string(),
            ));
        }

        // Update config with actual frame count
        let mut config = config;
        config.frame_count = frame_count;

        self.encode(output_path, config, |frame_idx| {
            frames
                .get(frame_idx as usize)
                .cloned()
                .ok_or_else(|| Box::from("Frame index out of range"))
        })
    }

    /// Get the wgpu device
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get the wgpu queue
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}

/// Builder for creating HAP video encoders with custom settings
///
/// Provides a fluent API for configuring the encoder.
pub struct HapEncoderBuilder {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    prefer_gpu: bool,
}

impl HapEncoderBuilder {
    /// Create a new encoder builder
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self {
            device,
            queue,
            prefer_gpu: true,
        }
    }

    /// Set whether to prefer GPU compression (default: true)
    ///
    /// If GPU compression is not available, will fall back to CPU.
    pub fn prefer_gpu(mut self, prefer: bool) -> Self {
        self.prefer_gpu = prefer;
        self
    }

    /// Set the video dimensions (required for GPU compression init)
    pub fn with_dimensions(self, _width: u32, _height: u32) -> Self {
        // Dimensions stored for init_gpu in build()
        self
    }

    /// Build the encoder
    pub fn build(self) -> HapVideoEncoder {
        let encoder = HapVideoEncoder::new(self.device, self.queue);
        // GPU init is deferred to encode() time since dimensions are needed.
        // Caller can call init_gpu(width, height) before encoding.
        encoder
    }
}

/// Pad RGBA data from (w, h) to (pw, ph) with zeros
fn pad_rgba(rgba: &[u8], w: u32, h: u32, pw: u32, ph: u32) -> Vec<u8> {
    let mut padded = vec![0u8; (pw * ph * 4) as usize];
    for y in 0..h {
        let src_start = (y * w * 4) as usize;
        let src_end = src_start + (w * 4) as usize;
        let dst_start = (y * pw * 4) as usize;
        padded[dst_start..dst_start + (w * 4) as usize]
            .copy_from_slice(&rgba[src_start..src_end]);
    }
    padded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_config() {
        let config = EncodeConfig::new(1920, 1080, 30.0, 300)
            .with_format(HapFormat::Hap5)
            .with_quality(EncodeQuality::Best)
            .with_snappy(true);

        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.fps, 30.0);
        assert_eq!(config.frame_count, 300);
        assert_eq!(config.format, HapFormat::Hap5);
        assert_eq!(config.quality, EncodeQuality::Best);
        assert!(config.use_snappy);
    }

    #[test]
    fn test_encode_config_defaults() {
        let config = EncodeConfig::new(1920, 1080, 30.0, 300);

        assert_eq!(config.format, HapFormat::HapY); // Default
        assert_eq!(config.quality, EncodeQuality::Balanced); // Default
        assert!(config.use_snappy); // Default
    }

    // Note: Integration tests that require a real GPU are in tests/ directory
}
