//! HAP Frame Encoder
//!
//! Encodes raw video frames to HAP format using DXT/BCn compression
//! and optional Snappy compression.

use hap_parser::TextureFormat;
use std::io;
use thiserror::Error;

/// Errors that can occur during HAP encoding
#[derive(Error, Debug)]
pub enum HapEncodeError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),

    #[error("Invalid pixel data: {0}")]
    InvalidPixelData(String),

    #[error("Compression error: {0}")]
    CompressionError(String),

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
}

/// HAP format variants
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HapFormat {
    /// DXT1 compression (RGB, 8:1 compression from RGBA)
    Hap1,
    /// DXT5 compression (RGBA, 4:1 compression)
    Hap5,
    /// YCoCg-DXT5 compression (High quality RGB, no alpha)
    HapY,
    /// BC4 compression (Alpha only)
    HapA,
    /// BC7 compression (High quality RGBA)
    Hap7,
    /// BC6H compression (HDR RGB)
    HapH,
}

impl HapFormat {
    /// Get the texture format for this HAP format
    pub fn texture_format(&self) -> TextureFormat {
        match self {
            HapFormat::Hap1 => TextureFormat::RgbDxt1,
            HapFormat::Hap5 => TextureFormat::RgbaDxt5,
            HapFormat::HapY => TextureFormat::YcoCgDxt5,
            HapFormat::HapA => TextureFormat::AlphaRgtc1,
            HapFormat::Hap7 => TextureFormat::RgbaBc7,
            HapFormat::HapH => TextureFormat::RgbBc6hUfloat,
        }
    }

    /// Get the format identifier byte for HAP header (uncompressed)
    /// Based on HAP specification
    pub fn uncompressed_identifier(&self) -> u8 {
        match self {
            HapFormat::Hap1 => 0xAB,  // DXT1, no compression
            HapFormat::Hap5 => 0xAE,  // DXT5, no compression
            HapFormat::HapY => 0xAF,  // YCoCg-DXT5, no compression
            HapFormat::HapA => 0xA1,  // BC4/RGTC1, no compression
            HapFormat::Hap7 => 0xAC,  // BC7, no compression
            HapFormat::HapH => 0xA2,  // BC6H unsigned, no compression
        }
    }

    /// Get the format identifier byte for HAP header (Snappy compressed)
    pub fn snappy_identifier(&self) -> u8 {
        match self {
            HapFormat::Hap1 => 0xBB,  // DXT1, Snappy
            HapFormat::Hap5 => 0xBE,  // DXT5, Snappy
            HapFormat::HapY => 0xBF,  // YCoCg-DXT5, Snappy
            HapFormat::HapA => 0xB1,  // BC4/RGTC1, Snappy
            HapFormat::Hap7 => 0xBC,  // BC7, Snappy
            HapFormat::HapH => 0xB2,  // BC6H unsigned, Snappy
        }
    }

    /// Get the bytes per block for this format
    pub fn bytes_per_block(&self) -> usize {
        match self {
            HapFormat::Hap1 => 8,   // DXT1: 8 bytes per 4x4 block
            HapFormat::Hap5 => 16,  // DXT5: 16 bytes per 4x4 block
            HapFormat::HapY => 16,  // YCoCg-DXT5: 16 bytes per 4x4 block
            HapFormat::HapA => 8,   // BC4: 8 bytes per 4x4 block
            HapFormat::Hap7 => 16,  // BC7: 16 bytes per 4x4 block
            HapFormat::HapH => 16,  // BC6H: 16 bytes per 4x4 block
        }
    }

    /// Get the four-character codec name for QuickTime
    pub fn codec_name(&self) -> &'static str {
        match self {
            HapFormat::Hap1 => "Hap1",
            HapFormat::Hap5 => "Hap5",
            HapFormat::HapY => "HapY",
            HapFormat::HapA => "HapA",
            HapFormat::Hap7 => "Hap7",
            HapFormat::HapH => "HapH",
        }
    }

    /// Check if this format has an alpha channel
    pub fn has_alpha(&self) -> bool {
        matches!(self, HapFormat::Hap5 | HapFormat::Hap7)
    }
}

/// Compression options for HAP encoding
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompressionMode {
    /// No second-stage compression (raw DXT data)
    None,
    /// Snappy compression (recommended)
    Snappy,
}

/// DXT compression quality preset
///
/// Controls the speed/quality tradeoff for CPU-based DXT compression.
/// For live performance encoding, use `Fast`. For offline/import encoding,
/// use `Balanced` or `Best`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DxtQuality {
    /// RangeFit: ~2x faster than Balanced, slightly lower quality.
    /// Recommended for live recording and real-time encoding.
    Fast,
    /// ClusterFit (default): good balance of speed and quality.
    Balanced,
    /// IterativeClusterFit: best quality, slowest.
    /// Recommended for final export / archival.
    Best,
}

impl Default for DxtQuality {
    fn default() -> Self {
        DxtQuality::Balanced
    }
}

#[cfg(feature = "cpu-compression")]
impl DxtQuality {
    fn to_texpresso_params(self) -> texpresso::Params {
        match self {
            DxtQuality::Fast => texpresso::Params {
                algorithm: texpresso::Algorithm::RangeFit,
                ..Default::default()
            },
            DxtQuality::Balanced => texpresso::Params::default(),
            DxtQuality::Best => texpresso::Params {
                algorithm: texpresso::Algorithm::IterativeClusterFit,
                ..Default::default()
            },
        }
    }
}

/// Encodes raw video frames to HAP format
pub struct HapFrameEncoder {
    format: HapFormat,
    width: u32,
    height: u32,
    compression: CompressionMode,
    quality: DxtQuality,
    padded_width: u32,
    padded_height: u32,
    blocks_x: u32,
    blocks_y: u32,
}

impl HapFrameEncoder {
    /// Create a new HAP frame encoder
    ///
    /// # Arguments
    ///
    /// * `format` - The HAP format variant to encode to
    /// * `width` - Frame width in pixels
    /// * `height` - Frame height in pixels
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are 0
    pub fn new(format: HapFormat, width: u32, height: u32) -> Result<Self, HapEncodeError> {
        if width == 0 || height == 0 {
            return Err(HapEncodeError::InvalidDimensions(
                "Width and height must be greater than 0".to_string()
            ));
        }

        // HAP requires dimensions to be padded to multiples of 4
        let padded_width = ((width + 3) / 4) * 4;
        let padded_height = ((height + 3) / 4) * 4;

        // Calculate number of DXT blocks
        let blocks_x = padded_width / 4;
        let blocks_y = padded_height / 4;

        Ok(Self {
            format,
            width,
            height,
            compression: CompressionMode::Snappy, // Default to Snappy for better compression
            quality: DxtQuality::default(),
            padded_width,
            padded_height,
            blocks_x,
            blocks_y,
        })
    }

    /// Set the DXT compression quality (default: Balanced)
    ///
    /// `Fast` uses RangeFit (~2x faster, recommended for live recording).
    /// `Balanced` uses ClusterFit (default texpresso quality).
    /// `Best` uses IterativeClusterFit (highest quality, slowest).
    pub fn set_quality(&mut self, quality: DxtQuality) {
        self.quality = quality;
    }

    /// Set the compression mode (default: Snappy)
    pub fn set_compression(&mut self, mode: CompressionMode) {
        self.compression = mode;
    }

    /// Get the HAP format
    pub fn format(&self) -> HapFormat {
        self.format
    }

    /// Get the frame dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Get the padded dimensions (multiple of 4)
    pub fn padded_dimensions(&self) -> (u32, u32) {
        (self.padded_width, self.padded_height)
    }

    /// Get the texture format
    pub fn texture_format(&self) -> TextureFormat {
        self.format.texture_format()
    }

    /// Calculate the expected DXT compressed size
    pub fn dxt_size(&self) -> usize {
        (self.blocks_x * self.blocks_y) as usize * self.format.bytes_per_block()
    }

    /// Encode RGBA pixels to HAP format
    ///
    /// # Arguments
    ///
    /// * `rgba_data` - Raw RGBA pixel data (width * height * 4 bytes)
    ///
    /// # Returns
    ///
    /// Encoded HAP frame data ready to be written to a container
    ///
    /// # Errors
    ///
    /// Returns an error if pixel data size is incorrect or compression fails
    pub fn encode(&self, rgba_data: &[u8]) -> Result<Vec<u8>, HapEncodeError> {
        let expected_size = (self.width * self.height * 4) as usize;
        if rgba_data.len() != expected_size {
            return Err(HapEncodeError::InvalidPixelData(format!(
                "Expected {} bytes for {}x{} RGBA, got {}",
                expected_size, self.width, self.height, rgba_data.len()
            )));
        }

        // Pad the input data if dimensions aren't multiples of 4
        let padded_data = if self.width != self.padded_width || self.height != self.padded_height {
            self.pad_rgba_data(rgba_data)
        } else {
            rgba_data.to_vec()
        };

        // Step 1: Compress to DXT format
        let dxt_data = self.compress_to_dxt(&padded_data)?;

        // Step 2: Apply Snappy compression + HAP header
        self.apply_compression_and_header(&dxt_data)
    }

    /// Encode pre-compressed DXT data into a HAP frame
    ///
    /// Applies Snappy compression (if enabled) and builds the HAP frame header.
    /// Use this when DXT compression is done externally (e.g., GPU compute shader).
    ///
    /// # Arguments
    ///
    /// * `dxt_data` - Raw DXT/BCn compressed texture data
    ///
    /// # Errors
    ///
    /// Returns an error if the DXT data size doesn't match the expected size
    /// for the configured format and dimensions.
    pub fn encode_from_dxt(&self, dxt_data: &[u8]) -> Result<Vec<u8>, HapEncodeError> {
        let expected_size = self.dxt_size();
        if dxt_data.len() != expected_size {
            return Err(HapEncodeError::InvalidPixelData(format!(
                "Expected {} bytes of DXT data for {}x{} {:?}, got {}",
                expected_size, self.padded_width, self.padded_height, self.format, dxt_data.len()
            )));
        }

        self.apply_compression_and_header(dxt_data)
    }

    /// Apply Snappy compression (if enabled) and build HAP frame header
    fn apply_compression_and_header(&self, dxt_data: &[u8]) -> Result<Vec<u8>, HapEncodeError> {
        let (compressed_data, is_compressed) = match self.compression {
            CompressionMode::Snappy => {
                let compressed = snap::raw::Encoder::new()
                    .compress_vec(dxt_data)
                    .map_err(|e| HapEncodeError::CompressionError(e.to_string()))?;

                if compressed.len() < dxt_data.len() {
                    (compressed, true)
                } else {
                    (dxt_data.to_vec(), false)
                }
            }
            CompressionMode::None => (dxt_data.to_vec(), false),
        };

        let header = self.build_header(is_compressed, compressed_data.len() as u32);

        let mut result = Vec::with_capacity(header.len() + compressed_data.len());
        result.extend_from_slice(&header);
        result.extend_from_slice(&compressed_data);

        Ok(result)
    }

    /// Encode RGBA pixels with explicit compression control
    ///
    /// This is an internal method that allows forcing compression on/off
    /// for testing and special cases.
    pub fn encode_with_compression(
        &self,
        rgba_data: &[u8],
        compression: CompressionMode,
    ) -> Result<Vec<u8>, HapEncodeError> {
        let encoder = Self {
            format: self.format,
            width: self.width,
            height: self.height,
            compression,
            quality: self.quality,
            padded_width: self.padded_width,
            padded_height: self.padded_height,
            blocks_x: self.blocks_x,
            blocks_y: self.blocks_y,
        };
        encoder.encode(rgba_data)
    }

    /// Pad RGBA data to multiple of 4 dimensions
    fn pad_rgba_data(&self, rgba_data: &[u8]) -> Vec<u8> {
        let padded_size = (self.padded_width * self.padded_height * 4) as usize;
        let mut padded = vec![0u8; padded_size];

        for y in 0..self.height {
            let src_start = (y * self.width * 4) as usize;
            let src_end = src_start + (self.width * 4) as usize;
            let dst_start = (y * self.padded_width * 4) as usize;
            
            padded[dst_start..dst_start + (self.width * 4) as usize]
                .copy_from_slice(&rgba_data[src_start..src_end]);
        }

        padded
    }

    /// Compress RGBA data to DXT format
    fn compress_to_dxt(&self, rgba_data: &[u8]) -> Result<Vec<u8>, HapEncodeError> {
        match self.format {
            HapFormat::Hap1 => self.compress_dxt1(rgba_data),
            HapFormat::Hap5 => self.compress_dxt5(rgba_data),
            HapFormat::HapY => self.compress_ycocg_dxt5(rgba_data),
            HapFormat::HapA => self.compress_bc4(rgba_data),
            HapFormat::Hap7 => self.compress_bc7(rgba_data),
            HapFormat::HapH => self.compress_bc6h(rgba_data),
        }
    }

    /// Compress RGBA data to DXT1 format
    #[cfg(feature = "cpu-compression")]
    fn compress_dxt1(&self, rgba_data: &[u8]) -> Result<Vec<u8>, HapEncodeError> {
        let block_count = (self.blocks_x * self.blocks_y) as usize;
        let mut output = vec![0u8; block_count * 8];

        // Compress entire image at once using texpresso
        // API: compress(input, width, height, params, output)
        texpresso::Format::Bc1.compress(
            rgba_data,
            self.padded_width as usize,
            self.padded_height as usize,
            self.quality.to_texpresso_params(),
            &mut output
        );

        Ok(output)
    }

    /// Compress RGBA data to DXT1 format (fallback without cpu-compression)
    #[cfg(not(feature = "cpu-compression"))]
    fn compress_dxt1(&self, _rgba_data: &[u8]) -> Result<Vec<u8>, HapEncodeError> {
        Err(HapEncodeError::UnsupportedFormat(
            "DXT1 compression requires 'cpu-compression' feature".to_string()
        ))
    }

    /// Compress RGBA data to DXT5 format
    #[cfg(feature = "cpu-compression")]
    fn compress_dxt5(&self, rgba_data: &[u8]) -> Result<Vec<u8>, HapEncodeError> {
        let block_count = (self.blocks_x * self.blocks_y) as usize;
        let mut output = vec![0u8; block_count * 16];

        // Compress entire image at once using texpresso
        texpresso::Format::Bc3.compress(
            rgba_data,
            self.padded_width as usize,
            self.padded_height as usize,
            self.quality.to_texpresso_params(),
            &mut output
        );

        Ok(output)
    }

    /// Compress RGBA data to DXT5 format (fallback without cpu-compression)
    #[cfg(not(feature = "cpu-compression"))]
    fn compress_dxt5(&self, _rgba_data: &[u8]) -> Result<Vec<u8>, HapEncodeError> {
        Err(HapEncodeError::UnsupportedFormat(
            "DXT5 compression requires 'cpu-compression' feature".to_string()
        ))
    }

    /// Compress RGB data to YCoCg-DXT5 format
    #[cfg(feature = "cpu-compression")]
    fn compress_ycocg_dxt5(&self, rgba_data: &[u8]) -> Result<Vec<u8>, HapEncodeError> {
        let block_count = (self.blocks_x * self.blocks_y) as usize;
        
        // First convert entire image to YCoCg format
        let mut ycocg_data = vec![0u8; rgba_data.len()];
        for (src_chunk, dst_chunk) in rgba_data.chunks_exact(4).zip(ycocg_data.chunks_exact_mut(4)) {
            let r = src_chunk[0] as i32;
            let g = src_chunk[1] as i32;
            let b = src_chunk[2] as i32;
            let a = src_chunk[3];

            let y = ((r + 2*g + b) / 4) as u8;
            let co = ((r - b) / 2 + 128) as u8;
            let cg = ((-r + 2*g - b) / 4 + 128) as u8;

            dst_chunk[0] = co;
            dst_chunk[1] = cg;
            dst_chunk[2] = y;
            dst_chunk[3] = a;
        }

        // Compress entire image at once
        let mut output = vec![0u8; block_count * 16];
        texpresso::Format::Bc3.compress(
            &ycocg_data,
            self.padded_width as usize,
            self.padded_height as usize,
            self.quality.to_texpresso_params(),
            &mut output
        );

        Ok(output)
    }

    /// Compress RGB data to YCoCg-DXT5 format (fallback)
    #[cfg(not(feature = "cpu-compression"))]
    fn compress_ycocg_dxt5(&self, _rgba_data: &[u8]) -> Result<Vec<u8>, HapEncodeError> {
        Err(HapEncodeError::UnsupportedFormat(
            "YCoCg-DXT5 compression requires 'cpu-compression' feature".to_string()
        ))
    }

    /// Compress alpha data to BC4 format
    #[cfg(feature = "cpu-compression")]
    fn compress_bc4(&self, rgba_data: &[u8]) -> Result<Vec<u8>, HapEncodeError> {
        let block_count = (self.blocks_x * self.blocks_y) as usize;
        
        // Extract alpha channel from entire image
        let pixel_count = (self.padded_width * self.padded_height) as usize;
        let mut alpha_data = vec![0u8; pixel_count];
        for (i, alpha) in alpha_data.iter_mut().enumerate() {
            *alpha = rgba_data[i * 4 + 3];
        }

        // Compress entire alpha image at once
        let mut output = vec![0u8; block_count * 8];
        texpresso::Format::Bc4.compress(
            &alpha_data,
            self.padded_width as usize,
            self.padded_height as usize,
            self.quality.to_texpresso_params(),
            &mut output
        );

        Ok(output)
    }

    /// Compress alpha data to BC4 format (fallback)
    #[cfg(not(feature = "cpu-compression"))]
    fn compress_bc4(&self, _rgba_data: &[u8]) -> Result<Vec<u8>, HapEncodeError> {
        Err(HapEncodeError::UnsupportedFormat(
            "BC4 compression requires 'cpu-compression' feature".to_string()
        ))
    }

    /// Compress RGBA data to BC7 format
    #[cfg(feature = "cpu-compression")]
    fn compress_bc7(&self, _rgba_data: &[u8]) -> Result<Vec<u8>, HapEncodeError> {
        // BC7 compression is complex and not supported by texpresso
        // Return an error for now - BC7 would need a specialized encoder
        Err(HapEncodeError::UnsupportedFormat(
            "BC7 compression not yet implemented".to_string()
        ))
    }

    /// Compress RGBA data to BC7 format (fallback)
    #[cfg(not(feature = "cpu-compression"))]
    fn compress_bc7(&self, _rgba_data: &[u8]) -> Result<Vec<u8>, HapEncodeError> {
        Err(HapEncodeError::UnsupportedFormat(
            "BC7 compression requires 'cpu-compression' feature".to_string()
        ))
    }

    /// Compress RGB data to BC6H format
    #[cfg(feature = "cpu-compression")]
    fn compress_bc6h(&self, _rgba_data: &[u8]) -> Result<Vec<u8>, HapEncodeError> {
        // BC6H compression is for HDR content and very complex
        Err(HapEncodeError::UnsupportedFormat(
            "BC6H compression not yet implemented".to_string()
        ))
    }

    /// Compress RGB data to BC6H format (fallback)
    #[cfg(not(feature = "cpu-compression"))]
    fn compress_bc6h(&self, _rgba_data: &[u8]) -> Result<Vec<u8>, HapEncodeError> {
        Err(HapEncodeError::UnsupportedFormat(
            "BC6H compression requires 'cpu-compression' feature".to_string()
        ))
    }

    /// Build HAP frame header according to HAP spec
    ///
    /// Simple frames (no second-stage compression):
    /// - 4-byte header: [size: 3 bytes LE][type: 1 byte]
    ///   where type is 0xAB, 0xAE, 0xAF, etc. (uncompressed types)
    ///
    /// Snappy-compressed frames:
    /// - 4-byte header: [size: 3 bytes LE][type: 1 byte]
    ///   where type is 0xBB, 0xBE, 0xBF, etc. (Snappy types)
    /// - Followed by Snappy-compressed data
    ///
    /// The size field is the size of the section data (after header)
    fn build_header(&self, is_compressed: bool, data_size: u32) -> Vec<u8> {
        let type_byte = if is_compressed {
            self.format.snappy_identifier()
        } else {
            self.format.uncompressed_identifier()
        };

        let mut header = Vec::with_capacity(4);
        
        // Standard 4-byte header: [size: 3 bytes LE][type: 1 byte]
        if data_size < 0x00FFFFFF {
            header.push((data_size & 0xFF) as u8);
            header.push(((data_size >> 8) & 0xFF) as u8);
            header.push(((data_size >> 16) & 0xFF) as u8);
            header.push(type_byte);
        } else {
            // Extended 8-byte header: [0,0,0][type][size: 4 bytes LE]
            header.push(0);
            header.push(0);
            header.push(0);
            header.push(type_byte);
            header.extend_from_slice(&data_size.to_le_bytes());
        }

        header
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hap_format_properties() {
        assert_eq!(HapFormat::Hap1.bytes_per_block(), 8);
        assert_eq!(HapFormat::Hap5.bytes_per_block(), 16);
        assert_eq!(HapFormat::HapY.bytes_per_block(), 16);
        
        assert!(!HapFormat::Hap1.has_alpha());
        assert!(HapFormat::Hap5.has_alpha());
        assert!(!HapFormat::HapY.has_alpha());
    }

    #[test]
    fn test_encoder_creation() {
        let encoder = HapFrameEncoder::new(HapFormat::Hap1, 1920, 1080);
        assert!(encoder.is_ok());
        
        let encoder = encoder.unwrap();
        assert_eq!(encoder.dimensions(), (1920, 1080));
        assert_eq!(encoder.padded_dimensions(), (1920, 1080)); // Already multiple of 4
    }

    #[test]
    fn test_encoder_padded_dimensions() {
        let encoder = HapFrameEncoder::new(HapFormat::Hap1, 1920, 1081).unwrap();
        // 1081 should be padded to 1084 (next multiple of 4)
        assert_eq!(encoder.padded_dimensions(), (1920, 1084));
    }

    #[test]
    fn test_encoder_invalid_dimensions() {
        let encoder = HapFrameEncoder::new(HapFormat::Hap1, 0, 1080);
        assert!(encoder.is_err());
        
        let encoder = HapFrameEncoder::new(HapFormat::Hap1, 1920, 0);
        assert!(encoder.is_err());
    }

    #[test]
    #[cfg(feature = "cpu-compression")]
    fn test_encode_small_frame() {
        // Test encoding a small 4x4 frame (minimum DXT block size)
        let encoder = HapFrameEncoder::new(HapFormat::Hap1, 4, 4).unwrap();
        
        // Create simple RGBA data (4x4 pixels * 4 bytes)
        let rgba_data = vec![255u8; 64];
        
        let result = encoder.encode(&rgba_data);
        assert!(result.is_ok());
        
        let encoded = result.unwrap();
        // Should have header (4-8 bytes) + compressed data
        assert!(!encoded.is_empty());
        
        // Verify header starts with valid Hap1 format identifier
        // Could be uncompressed (0xAB) or Snappy (0xBB) depending on compression ratio
        let format_byte = encoded[3];
        assert!(
            format_byte == HapFormat::Hap1.uncompressed_identifier() || 
            format_byte == HapFormat::Hap1.snappy_identifier(),
            "Invalid format byte: 0x{:02X}", format_byte
        );
    }

    #[test]
    #[cfg(feature = "cpu-compression")]
    fn test_encode_without_compression() {
        let encoder = HapFrameEncoder::new(HapFormat::Hap1, 4, 4).unwrap();
        
        let rgba_data = vec![255u8; 64];
        
        let encoded = encoder.encode_with_compression(&rgba_data, CompressionMode::None).unwrap();
        
        // Without compression, header should indicate uncompressed
        let format_byte = encoded[3];
        assert_eq!(format_byte, HapFormat::Hap1.uncompressed_identifier()); // Hap1 uncompressed
    }

    #[test]
    fn test_dxt_size_calculation() {
        // 4x4 pixels = 1 block * 8 bytes = 8 bytes for DXT1
        let encoder = HapFrameEncoder::new(HapFormat::Hap1, 4, 4).unwrap();
        assert_eq!(encoder.dxt_size(), 8);
        
        // 8x8 pixels = 4 blocks * 8 bytes = 32 bytes for DXT1
        let encoder = HapFrameEncoder::new(HapFormat::Hap1, 8, 8).unwrap();
        assert_eq!(encoder.dxt_size(), 32);
        
        // 4x4 pixels = 1 block * 16 bytes = 16 bytes for DXT5
        let encoder = HapFrameEncoder::new(HapFormat::Hap5, 4, 4).unwrap();
        assert_eq!(encoder.dxt_size(), 16);
    }
}
