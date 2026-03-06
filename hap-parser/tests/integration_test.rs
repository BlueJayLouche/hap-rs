//! Integration tests for hap-parser

use hap_parser::*;

/// Create a synthetic uncompressed HAP frame
fn create_test_frame_dxt1() -> Vec<u8> {
    let mut data = Vec::new();
    
    // Section header (4 bytes)
    // Size = 8 (1 block of DXT1 = 8 bytes)
    // Type = 0xAB (RGB DXT1, no compression)
    data.extend_from_slice(&[0x08, 0x00, 0x00, 0xAB]);
    
    // Texture data (8 bytes for one 4x4 DXT1 block)
    data.extend_from_slice(&[0u8; 8]);
    
    data
}

/// Create a synthetic Snappy-compressed HAP frame
fn create_test_frame_snappy() -> Vec<u8> {
    let mut data = Vec::new();
    
    // Texture data (uncompressed)
    let uncompressed = vec![0u8; 16]; // One DXT5 block
    
    // Compress with Snappy
    let compressed = snap::raw::Encoder::new()
        .compress_vec(&uncompressed)
        .expect("Failed to compress");
    
    // Section header (4 bytes)
    // Size = compressed.len()
    // Type = 0xBE (RGBA DXT5, Snappy)
    let size_bytes = (compressed.len() as u32).to_le_bytes();
    data.push(size_bytes[0]);
    data.push(size_bytes[1]);
    data.push(size_bytes[2]);
    data.push(0xBE); // Type
    
    // Compressed data
    data.extend_from_slice(&compressed);
    
    data
}

#[test]
fn test_parse_simple_dxt1_frame() {
    let frame_data = create_test_frame_dxt1();
    let frame = parse_frame(&frame_data).expect("Failed to parse frame");
    
    assert_eq!(frame.top_level_type, TopLevelType::RgbDxt1None);
    assert_eq!(frame.texture_format, TextureFormat::RgbDxt1);
    assert!(!frame.uses_snappy);
    assert_eq!(frame.texture_data.len(), 8);
}

#[test]
fn test_parse_snappy_dxt5_frame() {
    let frame_data = create_test_frame_snappy();
    let frame = parse_frame(&frame_data).expect("Failed to parse frame");
    
    assert_eq!(frame.top_level_type, TopLevelType::RgbaDxt5Snappy);
    assert_eq!(frame.texture_format, TextureFormat::RgbaDxt5);
    assert!(frame.uses_snappy);
    assert_eq!(frame.texture_data.len(), 16); // Decompressed size
}

#[test]
fn test_expected_size_calculations() {
    // DXT1: 8 bytes per 4x4 block
    assert_eq!(expected_texture_size(TextureFormat::RgbDxt1, 4, 4), 8);
    assert_eq!(expected_texture_size(TextureFormat::RgbDxt1, 8, 8), 32);
    
    // DXT5: 16 bytes per 4x4 block
    assert_eq!(expected_texture_size(TextureFormat::RgbaDxt5, 4, 4), 16);
    assert_eq!(expected_texture_size(TextureFormat::RgbaDxt5, 8, 8), 64);
    
    // Test with non-multiple-of-4 dimensions (should round up)
    // 5x5 rounds up to 8x8 = 2x2 blocks = 4 blocks
    assert_eq!(expected_texture_size(TextureFormat::RgbDxt1, 5, 5), 32); // 4 blocks * 8 bytes
    // 9x9 rounds up to 12x12 = 3x3 blocks = 9 blocks
    assert_eq!(expected_texture_size(TextureFormat::RgbDxt1, 9, 9), 72); // 9 blocks * 8 bytes
}

#[test]
fn test_all_top_level_types() {
    let test_cases = vec![
        (0xAB, TopLevelType::RgbDxt1None, TextureFormat::RgbDxt1, false),
        (0xBB, TopLevelType::RgbDxt1Snappy, TextureFormat::RgbDxt1, true),
        (0xAE, TopLevelType::RgbaDxt5None, TextureFormat::RgbaDxt5, false),
        (0xBE, TopLevelType::RgbaDxt5Snappy, TextureFormat::RgbaDxt5, true),
        (0xAF, TopLevelType::YcoCgDxt5None, TextureFormat::YcoCgDxt5, false),
        (0xBF, TopLevelType::YcoCgDxt5Snappy, TextureFormat::YcoCgDxt5, true),
    ];
    
    for (type_byte, expected_type, expected_format, expected_snappy) in test_cases {
        let parsed = TopLevelType::from_u8(type_byte)
            .expect(&format!("Should parse type 0x{:02X}", type_byte));
        
        assert_eq!(parsed, expected_type, "Type mismatch for 0x{:02X}", type_byte);
        assert_eq!(parsed.texture_format(), expected_format, "Format mismatch for 0x{:02X}", type_byte);
        assert_eq!(parsed.is_snappy(), expected_snappy, "Snappy mismatch for 0x{:02X}", type_byte);
    }
}

/// Create a synthetic complex HAP frame with 2 chunks
/// This mimics what ffmpeg produces with `-chunks 2`
fn create_test_frame_complex() -> Vec<u8> {
    let mut data = Vec::new();
    
    // Uncompressed texture data (2 DXT1 blocks = 16 bytes)
    let chunk0_data = vec![0xAAu8; 8]; // First block
    let chunk1_data = vec![0xBBu8; 8]; // Second block
    
    // Build Decode Instructions Container content (everything after the container header)
    let mut container_content = Vec::new();
    
    // 1. Compressor Table (section type 0x02) - 2 entries
    // Format: [compressor_byte] * chunk_count
    container_content.extend_from_slice(&[0x02, 0x00, 0x00, 0x02]); // Size=2, Type=0x02
    container_content.push(0x0A); // Chunk 0: No compression
    container_content.push(0x0A); // Chunk 1: No compression
    
    // 2. Chunk Size Table (section type 0x03) - 4 bytes per entry
    container_content.extend_from_slice(&[0x08, 0x00, 0x00, 0x03]); // Size=8, Type=0x03
    container_content.extend_from_slice(&(chunk0_data.len() as u32).to_le_bytes());
    container_content.extend_from_slice(&(chunk1_data.len() as u32).to_le_bytes());
    
    // 3. Chunk Offset Table (section type 0x04) - optional, 4 bytes per entry
    let offset0: u32 = 0;
    let offset1: u32 = chunk0_data.len() as u32;
    container_content.extend_from_slice(&[0x08, 0x00, 0x00, 0x04]); // Size=8, Type=0x04
    container_content.extend_from_slice(&offset0.to_le_bytes());
    container_content.extend_from_slice(&offset1.to_le_bytes());
    
    // Calculate sizes
    let container_content_size = container_content.len() as u32; // Size of content after container header
    let frame_data_size = (chunk0_data.len() + chunk1_data.len()) as u32;
    let section_size = 4 + container_content_size + frame_data_size; // Container header + content + frame data
    
    // Top-level section header
    // Type = 0xCB (RGB DXT1, Complex/Decode Instructions)
    data.extend_from_slice(&section_size.to_le_bytes()[0..3]);
    data.push(0xCB); // Type = Complex DXT1
    
    // Container header (4 bytes)
    // Type = 0x01 (Decode Instructions Container)
    // Size = content size (not including this header)
    data.extend_from_slice(&container_content_size.to_le_bytes()[0..3]);
    data.push(0x01); // Type = Decode Instructions Container
    
    // Container content
    data.extend_from_slice(&container_content);
    
    // Frame data (the actual chunks concatenated)
    data.extend_from_slice(&chunk0_data);
    data.extend_from_slice(&chunk1_data);
    
    data
}

#[test]
fn test_parse_complex_frame() {
    let frame_data = create_test_frame_complex();
    let frame = parse_frame(&frame_data).expect("Failed to parse complex frame");
    
    assert_eq!(frame.top_level_type, TopLevelType::RgbDxt1Complex);
    assert_eq!(frame.texture_format, TextureFormat::RgbDxt1);
    assert!(!frame.uses_snappy); // Chunks are uncompressed
    
    // Should have 2 chunks
    assert_eq!(frame.chunks.len(), 2, "Expected 2 chunks");
    
    // Check chunk info
    assert_eq!(frame.chunks[0].compressor, Compressor::None);
    assert_eq!(frame.chunks[0].size, 8);
    assert_eq!(frame.chunks[0].offset, 0);
    
    assert_eq!(frame.chunks[1].compressor, Compressor::None);
    assert_eq!(frame.chunks[1].size, 8);
    assert_eq!(frame.chunks[1].offset, 8);
    
    // Check decompressed texture data
    assert_eq!(frame.texture_data.len(), 16); // 2 blocks * 8 bytes
    assert_eq!(&frame.texture_data[0..8], &[0xAAu8; 8]);
    assert_eq!(&frame.texture_data[8..16], &[0xBBu8; 8]);
}

/// Create a complex frame with Snappy-compressed chunks
fn create_test_frame_complex_snappy() -> Vec<u8> {
    let mut data = Vec::new();
    
    // Compress chunk data with Snappy
    let chunk0_uncompressed = vec![0xCCu8; 16]; // DXT5 block
    let chunk1_uncompressed = vec![0xDDu8; 16]; // DXT5 block
    
    let chunk0_compressed = snap::raw::Encoder::new()
        .compress_vec(&chunk0_uncompressed)
        .expect("Failed to compress chunk 0");
    let chunk1_compressed = snap::raw::Encoder::new()
        .compress_vec(&chunk1_uncompressed)
        .expect("Failed to compress chunk 1");
    
    // Build Decode Instructions Container content (everything after the container header)
    let mut container_content = Vec::new();
    
    // 1. Compressor Table (section type 0x02)
    container_content.extend_from_slice(&[0x02, 0x00, 0x00, 0x02]); // Size=2, Type=0x02
    container_content.push(0x0B); // Chunk 0: Snappy
    container_content.push(0x0B); // Chunk 1: Snappy
    
    // 2. Chunk Size Table (section type 0x03)
    container_content.extend_from_slice(&[0x08, 0x00, 0x00, 0x03]); // Size=8, Type=0x03
    container_content.extend_from_slice(&(chunk0_compressed.len() as u32).to_le_bytes());
    container_content.extend_from_slice(&(chunk1_compressed.len() as u32).to_le_bytes());
    
    // Calculate sizes
    let container_content_size = container_content.len() as u32; // Size of content after container header
    let frame_data_size = (chunk0_compressed.len() + chunk1_compressed.len()) as u32;
    let section_size = 4 + container_content_size + frame_data_size; // Container header + content + frame data
    
    // Top-level section header
    // Type = 0xCE (RGBA DXT5, Complex/Decode Instructions)
    data.extend_from_slice(&section_size.to_le_bytes()[0..3]);
    data.push(0xCE); // Type = Complex DXT5
    
    // Container header (4 bytes)
    // Size = content size (not including this header)
    data.extend_from_slice(&container_content_size.to_le_bytes()[0..3]);
    data.push(0x01); // Type = Decode Instructions Container
    
    // Container content
    data.extend_from_slice(&container_content);
    
    // Frame data (compressed chunks concatenated - no offset table needed since sequential)
    data.extend_from_slice(&chunk0_compressed);
    data.extend_from_slice(&chunk1_compressed);
    
    data
}

#[test]
fn test_parse_complex_snappy_frame() {
    let frame_data = create_test_frame_complex_snappy();
    let frame = parse_frame(&frame_data).expect("Failed to parse complex Snappy frame");
    
    assert_eq!(frame.top_level_type, TopLevelType::RgbaDxt5Complex);
    assert_eq!(frame.texture_format, TextureFormat::RgbaDxt5);
    assert!(frame.uses_snappy); // Chunks are Snappy compressed
    
    // Should have 2 chunks
    assert_eq!(frame.chunks.len(), 2, "Expected 2 chunks");
    
    // Check chunk info
    assert_eq!(frame.chunks[0].compressor, Compressor::Snappy);
    assert_eq!(frame.chunks[1].compressor, Compressor::Snappy);
    
    // Check decompressed texture data
    assert_eq!(frame.texture_data.len(), 32); // 2 blocks * 16 bytes
    assert_eq!(&frame.texture_data[0..16], &[0xCCu8; 16]);
    assert_eq!(&frame.texture_data[16..32], &[0xDDu8; 16]);
}
