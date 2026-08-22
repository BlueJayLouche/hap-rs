//! Integration tests for hap-parser — synthetic frames mirroring what ffmpeg
//! and the reference HAP encoder produce.

use hap_parser::*;

/// Uncompressed DXT1 frame: one 4×4 block.
fn frame_dxt1() -> Vec<u8> {
    let mut data = vec![0x08, 0x00, 0x00, 0xAB]; // len=8, type=0xAB (DXT1, none)
    data.extend_from_slice(&[0u8; 8]);
    data
}

/// Snappy-compressed DXT5 frame: one 4×4 block.
fn frame_snappy() -> Vec<u8> {
    let compressed = snap::raw::Encoder::new().compress_vec(&[0u8; 16]).unwrap();
    let size = (compressed.len() as u32).to_le_bytes();
    let mut data = vec![size[0], size[1], size[2], 0xBE]; // type=0xBE (DXT5, Snappy)
    data.extend_from_slice(&compressed);
    data
}

#[test]
fn parse_simple_dxt1_frame() {
    let frame = parse_frame(&frame_dxt1()).unwrap();
    assert_eq!(frame.format, TextureFormat::RgbDxt1);
    assert_eq!(frame.data.len(), 8);
    assert!(frame.alpha.is_none());
}

#[test]
fn parse_snappy_dxt5_frame() {
    let frame = parse_frame(&frame_snappy()).unwrap();
    assert_eq!(frame.format, TextureFormat::RgbaDxt5);
    assert_eq!(frame.data.len(), 16); // decompressed
}

#[test]
fn frame_size_calculations() {
    assert_eq!(TextureFormat::RgbDxt1.frame_size(4, 4), 8);
    assert_eq!(TextureFormat::RgbDxt1.frame_size(8, 8), 32);
    assert_eq!(TextureFormat::RgbaDxt5.frame_size(4, 4), 16);
    assert_eq!(TextureFormat::RgbaDxt5.frame_size(8, 8), 64);
    // Non-multiple-of-4 rounds up to whole blocks.
    assert_eq!(TextureFormat::RgbDxt1.frame_size(5, 5), 32); // 2x2 blocks
    assert_eq!(TextureFormat::RgbDxt1.frame_size(9, 9), 72); // 3x3 blocks
}

#[test]
fn detect_format_matches_parse() {
    for (frame, expected) in [
        (frame_dxt1(), TextureFormat::RgbDxt1),
        (frame_snappy(), TextureFormat::RgbaDxt5),
    ] {
        assert_eq!(detect_format(&frame).unwrap(), expected);
        assert_eq!(parse_frame(&frame).unwrap().format, expected);
    }
}

/// Complex frame with an explicit chunk offset table (the path ffmpeg takes
/// with `-chunks N`). Two uncompressed DXT1 blocks.
fn frame_complex_with_offsets() -> Vec<u8> {
    let chunk0 = vec![0xAAu8; 8];
    let chunk1 = vec![0xBBu8; 8];

    let mut container = Vec::new();
    // Compressor table (0x02): both uncompressed (0x0A).
    container.extend_from_slice(&[0x02, 0x00, 0x00, 0x02, 0x0A, 0x0A]);
    // Size table (0x03): 4 bytes per chunk.
    container.extend_from_slice(&[0x08, 0x00, 0x00, 0x03]);
    container.extend_from_slice(&(chunk0.len() as u32).to_le_bytes());
    container.extend_from_slice(&(chunk1.len() as u32).to_le_bytes());
    // Offset table (0x04): explicit offsets into frame data.
    container.extend_from_slice(&[0x08, 0x00, 0x00, 0x04]);
    container.extend_from_slice(&0u32.to_le_bytes());
    container.extend_from_slice(&(chunk0.len() as u32).to_le_bytes());

    let container_len = container.len() as u32;
    let section_size = 4 + container_len + (chunk0.len() + chunk1.len()) as u32;

    let mut data = section_size.to_le_bytes()[0..3].to_vec();
    data.push(0xCB); // complex DXT1
    data.extend_from_slice(&container_len.to_le_bytes()[0..3]);
    data.push(0x01); // decode-instructions container
    data.extend_from_slice(&container);
    data.extend_from_slice(&chunk0);
    data.extend_from_slice(&chunk1);
    data
}

#[test]
fn parse_complex_frame_with_offset_table() {
    let frame = parse_frame(&frame_complex_with_offsets()).unwrap();
    assert_eq!(frame.format, TextureFormat::RgbDxt1);
    assert_eq!(frame.data.len(), 16);
    assert_eq!(&frame.data[0..8], &[0xAAu8; 8]);
    assert_eq!(&frame.data[8..16], &[0xBBu8; 8]);
}
