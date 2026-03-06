//! HAP Video Frame Parser
//!
//! This crate provides parsing for HAP video frames.
//! HAP is a GPU-accelerated codec that stores frames as compressed textures.

use byteorder::{ReadBytesExt, LE};
use std::io::{self, Read};
use thiserror::Error;

/// Errors that can occur during HAP parsing
#[derive(Error, Debug)]
pub enum HapError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    
    #[error("Invalid HAP frame: {0}")]
    InvalidFrame(String),
    
    #[error("Unknown texture format: 0x{0:02X}")]
    UnknownTextureFormat(u32),
    
    #[error("Unknown compressor type: 0x{0:02X}")]
    UnknownCompressor(u8),
    
    #[error("Snappy decompression failed: {0}")]
    SnappyError(String),
}

/// HAP texture formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureFormat {
    /// RGB DXT1/BC1
    RgbDxt1 = 0x83F0,
    /// RGBA DXT5/BC3
    RgbaDxt5 = 0x83F3,
    /// Scaled YCoCg DXT5
    YcoCgDxt5 = 0x01,
    /// Alpha RGTC1/BC4
    AlphaRgtc1 = 0x8DBB,
    /// RGBA BC7
    RgbaBc7 = 0x8E8C,
    /// RGB BC6H Unsigned Float
    RgbBc6hUfloat = 0x8E8F,
    /// RGB BC6H Signed Float
    RgbBc6hSfloat = 0x8E8E,
}

impl TextureFormat {
    /// Parse texture format from u32 value
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0x83F0 => Some(Self::RgbDxt1),
            0x83F3 => Some(Self::RgbaDxt5),
            0x01 => Some(Self::YcoCgDxt5),
            0x8DBB => Some(Self::AlphaRgtc1),
            0x8E8C => Some(Self::RgbaBc7),
            0x8E8F => Some(Self::RgbBc6hUfloat),
            0x8E8E => Some(Self::RgbBc6hSfloat),
            _ => None,
        }
    }
}

/// Second-stage compressor types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compressor {
    /// No compression
    None,
    /// Snappy compression
    Snappy,
}

impl Compressor {
    /// Parse compressor from u8 value
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0x0A => Some(Self::None),
            0x0B => Some(Self::Snappy),
            _ => None,
        }
    }
}

/// Top-level section types (legacy compatibility)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopLevelType {
    RgbDxt1None = 0xAB,
    RgbDxt1Snappy = 0xBB,
    RgbDxt1Complex = 0xCB,
    RgbaDxt5None = 0xAE,
    RgbaDxt5Snappy = 0xBE,
    RgbaDxt5Complex = 0xCE,
    YcoCgDxt5None = 0xAF,
    YcoCgDxt5Snappy = 0xBF,
    YcoCgDxt5Complex = 0xCF,
    RgbaBc7None = 0xAC,
    RgbaBc7Snappy = 0xBC,
    RgbaBc7Complex = 0xCC,
    AlphaRgtc1None = 0xA1,
    AlphaRgtc1Snappy = 0xB1,
    AlphaRgtc1Complex = 0xC1,
    RgbBc6hUfloatNone = 0xA2,
    RgbBc6hUfloatSnappy = 0xB2,
    RgbBc6hUfloatComplex = 0xC2,
    RgbBc6hSfloatNone = 0xA3,
    RgbBc6hSfloatSnappy = 0xB3,
    RgbBc6hSfloatComplex = 0xC3,
    MultipleImages = 0x0D,
}

impl TopLevelType {
    /// Parse from u8 value
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0xAB => Some(Self::RgbDxt1None),
            0xBB => Some(Self::RgbDxt1Snappy),
            0xCB => Some(Self::RgbDxt1Complex),
            0xAE => Some(Self::RgbaDxt5None),
            0xBE => Some(Self::RgbaDxt5Snappy),
            0xCE => Some(Self::RgbaDxt5Complex),
            0xAF => Some(Self::YcoCgDxt5None),
            0xBF => Some(Self::YcoCgDxt5Snappy),
            0xCF => Some(Self::YcoCgDxt5Complex),
            0xAC => Some(Self::RgbaBc7None),
            0xBC => Some(Self::RgbaBc7Snappy),
            0xCC => Some(Self::RgbaBc7Complex),
            0xA1 => Some(Self::AlphaRgtc1None),
            0xB1 => Some(Self::AlphaRgtc1Snappy),
            0xC1 => Some(Self::AlphaRgtc1Complex),
            0xA2 => Some(Self::RgbBc6hUfloatNone),
            0xB2 => Some(Self::RgbBc6hUfloatSnappy),
            0xC2 => Some(Self::RgbBc6hUfloatComplex),
            0xA3 => Some(Self::RgbBc6hSfloatNone),
            0xB3 => Some(Self::RgbBc6hSfloatSnappy),
            0xC3 => Some(Self::RgbBc6hSfloatComplex),
            0x0D => Some(Self::MultipleImages),
            _ => None,
        }
    }
    
    /// Get the texture format for this top-level type
    pub fn texture_format(&self) -> TextureFormat {
        match self {
            Self::RgbDxt1None | Self::RgbDxt1Snappy | Self::RgbDxt1Complex => TextureFormat::RgbDxt1,
            Self::RgbaDxt5None | Self::RgbaDxt5Snappy | Self::RgbaDxt5Complex => TextureFormat::RgbaDxt5,
            Self::YcoCgDxt5None | Self::YcoCgDxt5Snappy | Self::YcoCgDxt5Complex => TextureFormat::YcoCgDxt5,
            Self::RgbaBc7None | Self::RgbaBc7Snappy | Self::RgbaBc7Complex => TextureFormat::RgbaBc7,
            Self::AlphaRgtc1None | Self::AlphaRgtc1Snappy | Self::AlphaRgtc1Complex => TextureFormat::AlphaRgtc1,
            Self::RgbBc6hUfloatNone | Self::RgbBc6hUfloatSnappy | Self::RgbBc6hUfloatComplex => TextureFormat::RgbBc6hUfloat,
            Self::RgbBc6hSfloatNone | Self::RgbBc6hSfloatSnappy | Self::RgbBc6hSfloatComplex => TextureFormat::RgbBc6hSfloat,
            Self::MultipleImages => TextureFormat::RgbaDxt5,
        }
    }
    
    /// Check if this type uses Snappy compression
    pub fn is_snappy(&self) -> bool {
        matches!(self,
            Self::RgbDxt1Snappy | Self::RgbaDxt5Snappy | Self::YcoCgDxt5Snappy |
            Self::RgbaBc7Snappy | Self::AlphaRgtc1Snappy |
            Self::RgbBc6hUfloatSnappy | Self::RgbBc6hSfloatSnappy
        )
    }
    
    /// Check if this type uses decode instructions (complex)
    pub fn is_complex(&self) -> bool {
        matches!(self,
            Self::RgbDxt1Complex | Self::RgbaDxt5Complex | Self::YcoCgDxt5Complex |
            Self::RgbaBc7Complex | Self::AlphaRgtc1Complex |
            Self::RgbBc6hUfloatComplex | Self::RgbBc6hSfloatComplex | Self::MultipleImages
        )
    }
}

/// HAP section types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionType {
    // Top-level format types
    RgbDxt1None = 0xAB,
    RgbDxt1Snappy = 0xBB,
    RgbDxt1Complex = 0xCB,
    RgbaDxt5None = 0xAE,
    RgbaDxt5Snappy = 0xBE,
    RgbaDxt5Complex = 0xCE,
    YcoCgDxt5None = 0xAF,
    YcoCgDxt5Snappy = 0xBF,
    YcoCgDxt5Complex = 0xCF,
    RgbaBc7None = 0xAC,
    RgbaBc7Snappy = 0xBC,
    RgbaBc7Complex = 0xCC,
    AlphaRgtc1None = 0xA1,
    AlphaRgtc1Snappy = 0xB1,
    AlphaRgtc1Complex = 0xC1,
    RgbBc6hUfloatNone = 0xA2,
    RgbBc6hUfloatSnappy = 0xB2,
    RgbBc6hUfloatComplex = 0xC2,
    RgbBc6hSfloatNone = 0xA3,
    RgbBc6hSfloatSnappy = 0xB3,
    RgbBc6hSfloatComplex = 0xC3,
    
    // Container types
    MultipleImages = 0x0D,
    DecodeInstructionsContainer = 0x01,
    ChunkSecondStageCompressorTable = 0x02,
    ChunkSizeTable = 0x03,
    ChunkOffsetTable = 0x04,
}

/// Information about a chunk in a complex frame (legacy compatibility)
#[derive(Debug, Clone)]
pub struct ChunkInfo {
    /// Compressor type for this chunk
    pub compressor: Compressor,
    /// Chunk size in bytes
    pub size: u32,
    /// Chunk offset from start of frame data
    pub offset: u32,
}

/// A section within a HAP frame (legacy compatibility)
#[derive(Debug, Clone)]
pub struct Section {
    /// Section type byte
    pub section_type: u8,
    /// Raw section data (excluding header)
    pub data: Vec<u8>,
    /// Nested sections (for complex types)
    pub nested: Vec<Section>,
}

/// A parsed HAP frame
#[derive(Debug, Clone)]
pub struct HapFrame {
    /// Top-level section type (legacy compatibility)
    pub top_level_type: TopLevelType,
    /// Texture format
    pub texture_format: TextureFormat,
    /// Whether Snappy compression is used
    pub uses_snappy: bool,
    /// Raw compressed texture data (after decompression)
    pub texture_data: Vec<u8>,
    /// For complex frames: chunk information (legacy compatibility)
    pub chunks: Vec<ChunkInfo>,
}

/// Read a 3-byte little-endian unsigned integer
fn read_u24_le(data: &[u8]) -> u32 {
    (data[0] as u32) | ((data[1] as u32) << 8) | ((data[2] as u32) << 16)
}

/// Read a 4-byte little-endian unsigned integer
fn read_u32_le(data: &[u8]) -> u32 {
    (data[0] as u32) | ((data[1] as u32) << 8) | ((data[2] as u32) << 16) | ((data[3] as u32) << 24)
}

/// Parse a section header
/// Returns (section_length, section_type, header_length)
fn parse_section_header(data: &[u8]) -> Result<(u32, u8, usize), HapError> {
    if data.len() < 4 {
        return Err(HapError::InvalidFrame(
            format!("Section header too small: {} bytes", data.len())
        ));
    }
    
    // First 3 bytes are the size (little-endian) or 0 if using extended header
    let size = read_u24_le(&data[0..3]);
    let section_type = data[3];
    
    if size == 0 {
        // Extended 8-byte header
        if data.len() < 8 {
            return Err(HapError::InvalidFrame(
                "Extended header requires 8 bytes".to_string()
            ));
        }
        let size = read_u32_le(&data[4..8]);
        Ok((size, section_type, 8))
    } else {
        // Standard 4-byte header
        Ok((size, section_type, 4))
    }
}

/// Parse a HAP frame from raw bytes
pub fn parse_frame(data: &[u8]) -> Result<HapFrame, HapError> {
    if data.len() < 4 {
        return Err(HapError::InvalidFrame(
            format!("Frame data too small: {} bytes", data.len())
        ));
    }
    
    // Parse top-level section header
    let (section_length, section_type, header_length) = parse_section_header(data)?;
    
    // Get the top-level type
    let top_level_type = TopLevelType::from_u8(section_type)
        .ok_or_else(|| HapError::InvalidFrame(format!("Unknown top-level type: 0x{:02X}", section_type)))?;
    
    // Validate section length
    let total_length = header_length + section_length as usize;
    if total_length > data.len() {
        return Err(HapError::InvalidFrame(
            format!("Section extends beyond buffer: {} > {}", total_length, data.len())
        ));
    }
    
    // Get the section data (excluding header)
    let section_data = &data[header_length..total_length];
    
    // Get texture format
    let texture_format = top_level_type.texture_format();
    
    match top_level_type {
        t if t.is_snappy() => {
            // Snappy compression
            let decompressed = snap::raw::Decoder::new()
                .decompress_vec(section_data)
                .map_err(|e| HapError::SnappyError(e.to_string()))?;
            
            Ok(HapFrame {
                top_level_type,
                texture_format,
                uses_snappy: true,
                texture_data: decompressed,
                chunks: vec![],
            })
        }
        t if t.is_complex() => {
            // Complex - decode instructions container
            parse_complex_frame(top_level_type, texture_format, section_data)
        }
        _ => {
            // No compression - section data is raw texture data
            Ok(HapFrame {
                top_level_type,
                texture_format,
                uses_snappy: false,
                texture_data: section_data.to_vec(),
                chunks: vec![],
            })
        }
    }
}

/// Parse a complex frame with decode instructions
fn parse_complex_frame(top_level_type: TopLevelType, texture_format: TextureFormat, data: &[u8]) -> Result<HapFrame, HapError> {
    if data.len() < 4 {
        return Err(HapError::InvalidFrame(
            format!("Complex frame too small: {} bytes", data.len())
        ));
    }
    
    // Parse Decode Instructions Container header
    let (container_length, container_type, container_header_len) = parse_section_header(data)?;
    
    if container_type != 0x01 {
        return Err(HapError::InvalidFrame(
            format!("Expected Decode Instructions Container (0x01), got 0x{:02X}", container_type)
        ));
    }
    
    // Get container data
    let container_end = container_header_len + container_length as usize;
    if container_end > data.len() {
        return Err(HapError::InvalidFrame(
            format!("Container extends beyond data: {} > {}", container_end, data.len())
        ));
    }
    let container_data = &data[container_header_len..container_end];
    
    // Frame data starts after the container
    let frame_data = &data[container_end..];
    
    // Parse sections inside the container
    let mut compressor_table: Option<&[u8]> = None;
    let mut size_table: Option<&[u8]> = None;
    let mut offset_table: Option<&[u8]> = None;
    
    let mut pos = 0usize;
    while pos < container_data.len() {
        if container_data.len() - pos < 4 {
            break;
        }
        
        let (section_len, section_type, header_len) = parse_section_header(&container_data[pos..])?;
        let section_start = pos + header_len;
        let section_end = section_start + section_len as usize;
        
        if section_end > container_data.len() {
            return Err(HapError::InvalidFrame(
                "Section extends beyond container".to_string()
            ));
        }
        
        let section_data_slice = &container_data[section_start..section_end];
        
        match section_type {
            0x02 => compressor_table = Some(section_data_slice),
            0x03 => size_table = Some(section_data_slice),
            0x04 => offset_table = Some(section_data_slice),
            _ => {} // Ignore unknown sections
        }
        
        pos = section_end;
    }
    
    // Compressor table and size table are required
    let compressor_table = compressor_table
        .ok_or_else(|| HapError::InvalidFrame("Missing compressor table".to_string()))?;
    let size_table = size_table
        .ok_or_else(|| HapError::InvalidFrame("Missing size table".to_string()))?;
    
    let chunk_count = compressor_table.len();
    if chunk_count == 0 {
        return Err(HapError::InvalidFrame("No chunks in frame".to_string()));
    }
    
    // Validate size table has correct number of entries (4 bytes per entry)
    if size_table.len() != chunk_count * 4 {
        return Err(HapError::InvalidFrame(
            format!("Size table mismatch: expected {} bytes, got {}", chunk_count * 4, size_table.len())
        ));
    }
    
    // Build chunk metadata and decompress
    let mut chunks = Vec::with_capacity(chunk_count);
    let mut texture_data = Vec::new();
    let mut running_offset = 0usize;
    let mut has_snappy = false;
    
    for i in 0..chunk_count {
        let compressor = compressor_table[i];
        let chunk_size = read_u32_le(&size_table[i * 4..(i + 1) * 4]) as usize;
        
        // Get chunk offset
        let chunk_offset = if let Some(offset_table) = offset_table {
            if offset_table.len() < (i + 1) * 4 {
                return Err(HapError::InvalidFrame("Offset table too small".to_string()));
            }
            read_u32_le(&offset_table[i * 4..(i + 1) * 4]) as usize
        } else {
            running_offset
        };
        
        // Get chunk data from frame_data
        let chunk_end = chunk_offset + chunk_size;
        if chunk_end > frame_data.len() {
            return Err(HapError::InvalidFrame(
                format!("Chunk {} extends beyond frame data: {} > {}", i, chunk_end, frame_data.len())
            ));
        }
        let chunk_data = &frame_data[chunk_offset..chunk_end];
        
        // Determine compressor type
        let compressor_type = Compressor::from_u8(compressor)
            .ok_or_else(|| HapError::InvalidFrame(format!("Unknown chunk compressor: 0x{:02X}", compressor)))?;
        
        // Record chunk info
        let chunk_info = ChunkInfo {
            compressor: compressor_type,
            size: chunk_size as u32,
            offset: chunk_offset as u32,
        };
        chunks.push(chunk_info);
        
        // Decompress based on compressor type
        match compressor_type {
            Compressor::None => {
                texture_data.extend_from_slice(chunk_data);
            }
            Compressor::Snappy => {
                has_snappy = true;
                let decompressed = snap::raw::Decoder::new()
                    .decompress_vec(chunk_data)
                    .map_err(|e| HapError::SnappyError(format!("Chunk {}: {}", i, e)))?;
                texture_data.extend_from_slice(&decompressed);
            }
        }
        
        if offset_table.is_none() {
            running_offset += chunk_size;
        }
    }
    
    Ok(HapFrame {
        top_level_type,
        texture_format,
        uses_snappy: has_snappy,
        texture_data,
        chunks,
    })
}

/// Get the bytes-per-block for a texture format
pub fn bytes_per_block(format: TextureFormat) -> usize {
    match format {
        TextureFormat::RgbDxt1 => 8,
        TextureFormat::RgbaDxt5 | TextureFormat::YcoCgDxt5 => 16,
        TextureFormat::AlphaRgtc1 => 8,
        TextureFormat::RgbaBc7 => 16,
        TextureFormat::RgbBc6hUfloat | TextureFormat::RgbBc6hSfloat => 16,
    }
}

/// Calculate expected texture data size for given dimensions
pub fn expected_texture_size(format: TextureFormat, width: u32, height: u32) -> usize {
    let block_size = bytes_per_block(format);
    // DXT blocks are 4x4 pixels
    let blocks_x = ((width + 3) / 4) as usize;
    let blocks_y = ((height + 3) / 4) as usize;
    blocks_x * blocks_y * block_size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_texture_format_from_u32() {
        assert_eq!(TextureFormat::from_u32(0x83F0), Some(TextureFormat::RgbDxt1));
        assert_eq!(TextureFormat::from_u32(0x83F3), Some(TextureFormat::RgbaDxt5));
        assert_eq!(TextureFormat::from_u32(0x01), Some(TextureFormat::YcoCgDxt5));
        assert_eq!(TextureFormat::from_u32(0x9999), None);
    }

    #[test]
    fn test_bytes_per_block() {
        assert_eq!(bytes_per_block(TextureFormat::RgbDxt1), 8);
        assert_eq!(bytes_per_block(TextureFormat::RgbaDxt5), 16);
        assert_eq!(bytes_per_block(TextureFormat::YcoCgDxt5), 16);
        assert_eq!(bytes_per_block(TextureFormat::AlphaRgtc1), 8);
    }
    
    #[test]
    fn test_expected_texture_size() {
        // 1280x720 DXT5: (320 * 180) blocks * 16 bytes
        let size = expected_texture_size(TextureFormat::RgbaDxt5, 1280, 720);
        assert_eq!(size, 320 * 180 * 16);
        
        // 4x4 DXT1: 1 block * 8 bytes
        let size = expected_texture_size(TextureFormat::RgbDxt1, 4, 4);
        assert_eq!(size, 8);
    }
}
