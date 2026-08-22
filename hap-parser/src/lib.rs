//! HAP video frame parser.
//!
//! HAP is a GPU-accelerated video codec that stores each frame as a
//! block-compressed (BCn) texture, so frames upload straight to the GPU with no
//! CPU pixel decode. This crate parses one raw HAP frame (a single packet from a
//! QuickTime/MP4 container) into ready-to-upload texture data.
//!
//! ```no_run
//! # fn run(frame_bytes: &[u8]) -> Result<(), hap_parser::HapError> {
//! let frame = hap_parser::parse_frame(frame_bytes)?;
//! // frame.format -> which BC format, frame.data -> the BCn blocks.
//! if let Some(alpha) = &frame.alpha {
//!     // Hap Q Alpha: separate BC4 alpha plane.
//!     let _ = &alpha.data;
//! }
//! # Ok(()) }
//! ```
//!
//! # Frame layout
//!
//! A HAP frame is one or more sections. Each section has a header — 4 bytes
//! (`[len: u24 LE][type: u8]`) or, when the 3-byte length is zero, 8 bytes
//! (`[0,0,0][type][len: u32 LE]`). A top-level section's type byte encodes the
//! texture format in its low nibble and the compressor in its high nibble:
//!
//! | High nibble | Compressor      | Low nibble | Texture format        |
//! |-------------|-----------------|------------|-----------------------|
//! | `0xA`       | none            | `0xB`      | RGB DXT1 (BC1)        |
//! | `0xB`       | Snappy          | `0xE`      | RGBA DXT5 (BC3)       |
//! | `0xC`       | complex/chunked | `0xF`      | scaled YCoCg DXT5     |
//! |             |                 | `0xC`      | RGBA BC7              |
//! |             |                 | `0x1`      | alpha RGTC1 (BC4)     |
//!
//! Section type `0x0D` is a multi-image container (Hap Q Alpha): a color plane
//! plus a BC4 alpha plane, surfaced here as [`HapFrame::alpha`].

use thiserror::Error;

/// Errors produced while parsing a HAP frame.
#[derive(Error, Debug)]
pub enum HapError {
    /// A declared section length runs past the end of the buffer.
    #[error("HAP frame truncated: need {need} bytes, have {have}")]
    Truncated { need: usize, have: usize },

    /// The low nibble of a section type is not a known texture format.
    #[error("unknown HAP texture format: nibble 0x{0:X}")]
    UnknownTextureFormat(u8),

    /// The high nibble of a section type is not a known compressor.
    #[error("unknown HAP compressor: 0x{0:02X}")]
    UnknownCompressor(u8),

    /// A complex frame did not start with a decode-instructions section.
    #[error("expected decode-instructions section (0x01), got 0x{0:02X}")]
    UnexpectedSection(u8),

    /// A multi-image frame had no non-alpha (color) plane.
    #[error("multi-image HAP frame is missing its color plane")]
    MissingColorPlane,

    /// Snappy decompression failed.
    #[error("Snappy decompression failed: {0}")]
    Snappy(String),
}

/// GPU block-compressed texture format carried by a HAP frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureFormat {
    /// BC1 / DXT1 — RGB, no alpha (Hap).
    RgbDxt1,
    /// BC3 / DXT5 — RGBA with interpolated alpha (Hap Alpha).
    RgbaDxt5,
    /// BC3 / DXT5 carrying scaled YCoCg — needs YCoCg→RGB conversion (Hap Q).
    YcoCgDxt5,
    /// BC4 / RGTC1 — single-channel alpha (Hap Alpha-Only, or the alpha plane
    /// of Hap Q Alpha).
    AlphaRgtc1,
    /// BC7 / BPTC — RGBA, highest quality (Hap R).
    RgbaBc7,
}

impl TextureFormat {
    /// Bytes per 4×4 texel block.
    pub fn bytes_per_block(self) -> usize {
        match self {
            Self::RgbDxt1 | Self::AlphaRgtc1 => 8,
            Self::RgbaDxt5 | Self::YcoCgDxt5 | Self::RgbaBc7 => 16,
        }
    }

    /// Whether this format stores scaled YCoCg and therefore needs a YCoCg→RGB
    /// shader pass before display (Hap Q). All other formats are display-ready
    /// after GPU decode.
    pub fn needs_ycocg_convert(self) -> bool {
        matches!(self, Self::YcoCgDxt5)
    }

    /// Expected byte size of a full frame at the given pixel dimensions, with
    /// dimensions rounded up to whole 4×4 blocks.
    pub fn frame_size(self, width: u32, height: u32) -> usize {
        let blocks_x = width.div_ceil(4) as usize;
        let blocks_y = height.div_ceil(4) as usize;
        blocks_x * blocks_y * self.bytes_per_block()
    }

    fn from_section(section_type: u8) -> Result<Self, HapError> {
        match section_type & 0x0F {
            FMT_RGB_DXT1 => Ok(Self::RgbDxt1),
            FMT_RGBA_DXT5 => Ok(Self::RgbaDxt5),
            FMT_YCOCG_DXT5 => Ok(Self::YcoCgDxt5),
            FMT_ALPHA_RGTC1 => Ok(Self::AlphaRgtc1),
            FMT_RGBA_BC7 => Ok(Self::RgbaBc7),
            other => Err(HapError::UnknownTextureFormat(other)),
        }
    }
}

// Compressor — high nibble of a top-level / image section type.
const COMPRESSOR_NONE: u8 = 0xA0;
const COMPRESSOR_SNAPPY: u8 = 0xB0;
const COMPRESSOR_COMPLEX: u8 = 0xC0;

// Texture format — low nibble.
const FMT_ALPHA_RGTC1: u8 = 0x01;
const FMT_RGBA_BC7: u8 = 0x0C;
const FMT_RGB_DXT1: u8 = 0x0B;
const FMT_RGBA_DXT5: u8 = 0x0E;
const FMT_YCOCG_DXT5: u8 = 0x0F;

// Section types inside a frame.
const SECTION_MULTI_IMAGE: u8 = 0x0D;
const SECTION_DECODE_INSTRUCTIONS: u8 = 0x01;
const SECTION_CHUNK_COMPRESSORS: u8 = 0x02;
const SECTION_CHUNK_SIZES: u8 = 0x03;
const SECTION_CHUNK_OFFSETS: u8 = 0x04;

// Per-chunk second-stage compressor codes (inside a decode-instructions table).
const CHUNK_NONE: u8 = 0x0A;
const CHUNK_SNAPPY: u8 = 0x0B;

struct Header {
    section_type: u8,
    body_len: usize,
    header_len: usize,
}

/// Parse a section header and validate that its declared body fits in `data`.
fn parse_header(data: &[u8]) -> Result<Header, HapError> {
    if data.len() < 4 {
        return Err(HapError::Truncated { need: 4, have: data.len() });
    }
    let (body_len, header_len) = if data[0] == 0 && data[1] == 0 && data[2] == 0 {
        if data.len() < 8 {
            return Err(HapError::Truncated { need: 8, have: data.len() });
        }
        (u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize, 8)
    } else {
        ((data[0] as usize) | ((data[1] as usize) << 8) | ((data[2] as usize) << 16), 4)
    };
    if body_len > data.len() - header_len {
        return Err(HapError::Truncated { need: header_len + body_len, have: data.len() });
    }
    Ok(Header { section_type: data[3], body_len, header_len })
}

/// Body slice of a section. Safe: `parse_header` validated the bounds.
fn section_body<'a>(data: &'a [u8], h: &Header) -> &'a [u8] {
    &data[h.header_len..h.header_len + h.body_len]
}

fn snappy(src: &[u8]) -> Result<Vec<u8>, HapError> {
    let len = snap::raw::decompress_len(src).map_err(|e| HapError::Snappy(e.to_string()))?;
    let mut out = vec![0u8; len];
    snap::raw::Decoder::new()
        .decompress(src, &mut out)
        .map_err(|e| HapError::Snappy(e.to_string()))?;
    Ok(out)
}

/// Decode one image section (the texture format is its low nibble; the
/// compressor is its high nibble) into raw BCn data.
fn decode_section(section_type: u8, body: &[u8]) -> Result<(TextureFormat, Vec<u8>), HapError> {
    let format = TextureFormat::from_section(section_type)?;
    let data = match section_type & 0xF0 {
        COMPRESSOR_NONE => body.to_vec(),
        COMPRESSOR_SNAPPY => snappy(body)?,
        COMPRESSOR_COMPLEX => decode_chunked(body)?,
        other => return Err(HapError::UnknownCompressor(other)),
    };
    Ok((format, data))
}

/// Decode a complex (chunked) section: a decode-instructions container followed
/// by chunk data. Chunks are concatenated in table order.
fn decode_chunked(data: &[u8]) -> Result<Vec<u8>, HapError> {
    let h = parse_header(data)?;
    if h.section_type != SECTION_DECODE_INSTRUCTIONS {
        return Err(HapError::UnexpectedSection(h.section_type));
    }
    let instructions = section_body(data, &h);
    let frame_data = &data[h.header_len + h.body_len..];

    let (mut compressors, mut sizes, mut offsets): (&[u8], &[u8], Option<&[u8]>) = (&[], &[], None);
    let mut pos = 0;
    while pos < instructions.len() {
        let s = parse_header(&instructions[pos..])?;
        let d = section_body(&instructions[pos..], &s);
        match s.section_type {
            SECTION_CHUNK_COMPRESSORS => compressors = d,
            SECTION_CHUNK_SIZES => sizes = d,
            SECTION_CHUNK_OFFSETS => offsets = Some(d),
            _ => {}
        }
        pos += s.header_len + s.body_len;
    }

    let n = compressors.len();
    if sizes.len() < n * 4 {
        return Err(HapError::Truncated { need: n * 4, have: sizes.len() });
    }
    let read_u32 = |t: &[u8], i: usize| u32::from_le_bytes([t[i], t[i + 1], t[i + 2], t[i + 3]]) as usize;

    let mut out = Vec::new();
    let mut running = 0usize;
    for (i, &compressor) in compressors.iter().enumerate() {
        let size = read_u32(sizes, i * 4);
        let off = match offsets {
            Some(o) => {
                if o.len() < (i + 1) * 4 {
                    return Err(HapError::Truncated { need: (i + 1) * 4, have: o.len() });
                }
                read_u32(o, i * 4)
            }
            None => running,
        };
        let end = off.checked_add(size).filter(|&e| e <= frame_data.len()).ok_or(
            HapError::Truncated { need: off.saturating_add(size), have: frame_data.len() },
        )?;
        let chunk = &frame_data[off..end];
        match compressor {
            CHUNK_NONE => out.extend_from_slice(chunk),
            CHUNK_SNAPPY => out.extend_from_slice(&snappy(chunk)?),
            other => return Err(HapError::UnknownCompressor(other)),
        }
        running += size;
    }
    Ok(out)
}

/// A parsed HAP frame: decompressed BCn texture data ready for GPU upload.
#[derive(Debug, Clone)]
pub struct HapFrame {
    /// Format of the color (or sole) plane.
    pub format: TextureFormat,
    /// Decompressed BCn block data for the color/sole plane.
    pub data: Vec<u8>,
    /// Second plane for Hap Q Alpha (dual-plane). `None` for every other variant.
    pub alpha: Option<AlphaPlane>,
}

/// The separate alpha plane of a dual-plane Hap Q Alpha frame.
#[derive(Debug, Clone)]
pub struct AlphaPlane {
    /// Plane format — [`TextureFormat::AlphaRgtc1`] (BC4) in practice.
    pub format: TextureFormat,
    /// Decompressed BC4 block data.
    pub data: Vec<u8>,
}

/// Parse a raw HAP frame into decompressed, GPU-ready texture data.
pub fn parse_frame(data: &[u8]) -> Result<HapFrame, HapError> {
    let h = parse_header(data)?;
    let frame_body = section_body(data, &h);

    if h.section_type != SECTION_MULTI_IMAGE {
        let (format, data) = decode_section(h.section_type, frame_body)?;
        return Ok(HapFrame { format, data, alpha: None });
    }

    // Multi-image (Hap Q Alpha): a color plane plus a BC4 alpha plane.
    let mut color: Option<(TextureFormat, Vec<u8>)> = None;
    let mut alpha: Option<AlphaPlane> = None;
    let mut pos = 0;
    while pos < frame_body.len() {
        let s = parse_header(&frame_body[pos..])?;
        let sub = section_body(&frame_body[pos..], &s);
        let (format, bytes) = decode_section(s.section_type, sub)?;
        if format == TextureFormat::AlphaRgtc1 {
            alpha = Some(AlphaPlane { format, data: bytes });
        } else {
            color = Some((format, bytes));
        }
        pos += s.header_len + s.body_len;
    }
    let (format, data) = color.ok_or(HapError::MissingColorPlane)?;
    Ok(HapFrame { format, data, alpha })
}

/// Read just the color-plane texture format of a frame, reading only section
/// headers — no decompression. Use this to size GPU textures and staging
/// buffers before decoding, since container metadata alone does not distinguish
/// HAP variants (ffmpeg maps them all to one codec id).
pub fn detect_format(data: &[u8]) -> Result<TextureFormat, HapError> {
    let h = parse_header(data)?;
    if h.section_type != SECTION_MULTI_IMAGE {
        return TextureFormat::from_section(h.section_type);
    }
    let frame_body = section_body(data, &h);
    let mut pos = 0;
    while pos < frame_body.len() {
        let s = parse_header(&frame_body[pos..])?;
        let fmt = TextureFormat::from_section(s.section_type)?;
        if fmt != TextureFormat::AlphaRgtc1 {
            return Ok(fmt);
        }
        pos += s.header_len + s.body_len;
    }
    Err(HapError::MissingColorPlane)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a section with a 4-byte header.
    fn section(section_type: u8, payload: &[u8]) -> Vec<u8> {
        let len = payload.len();
        assert!(len < 0x00FF_FFFF && len > 0, "use section_long for 0 or >=16MB");
        let mut buf = vec![len as u8, (len >> 8) as u8, (len >> 16) as u8, section_type];
        buf.extend_from_slice(payload);
        buf
    }

    /// Build a section with an 8-byte (extended) header.
    fn section_long(section_type: u8, payload: &[u8]) -> Vec<u8> {
        let mut buf = vec![0, 0, 0, section_type];
        buf.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        buf.extend_from_slice(payload);
        buf
    }

    #[test]
    fn header_4byte() {
        let h = parse_header(&section(0xAB, &[1, 2, 3, 4, 5])).unwrap();
        assert_eq!((h.section_type, h.body_len, h.header_len), (0xAB, 5, 4));
    }

    #[test]
    fn header_8byte() {
        let h = parse_header(&section_long(0xAE, &[10, 20, 30])).unwrap();
        assert_eq!((h.section_type, h.body_len, h.header_len), (0xAE, 3, 8));
    }

    #[test]
    fn header_too_short_errors() {
        assert!(matches!(parse_header(&[1, 2]), Err(HapError::Truncated { .. })));
    }

    #[test]
    fn header_overruns_buffer_errors() {
        // Claims 99-byte body but only 1 byte follows.
        let bad = vec![99, 0, 0, 0xAB, 0x00];
        assert!(matches!(parse_header(&bad), Err(HapError::Truncated { .. })));
    }

    #[test]
    fn format_from_each_nibble() {
        assert_eq!(TextureFormat::from_section(0xAB).unwrap(), TextureFormat::RgbDxt1);
        assert_eq!(TextureFormat::from_section(0xBE).unwrap(), TextureFormat::RgbaDxt5);
        assert_eq!(TextureFormat::from_section(0xCF).unwrap(), TextureFormat::YcoCgDxt5);
        assert_eq!(TextureFormat::from_section(0xAC).unwrap(), TextureFormat::RgbaBc7);
        assert_eq!(TextureFormat::from_section(0xB1).unwrap(), TextureFormat::AlphaRgtc1);
        assert!(TextureFormat::from_section(0xA9).is_err());
    }

    #[test]
    fn bytes_per_block_and_frame_size() {
        assert_eq!(TextureFormat::RgbDxt1.bytes_per_block(), 8);
        assert_eq!(TextureFormat::RgbaBc7.bytes_per_block(), 16);
        // 1280x720 DXT5 = 320*180 blocks * 16 bytes.
        assert_eq!(TextureFormat::RgbaDxt5.frame_size(1280, 720), 320 * 180 * 16);
        // Non-multiple-of-4 rounds up: 5x5 BC1 = 2*2 blocks * 8.
        assert_eq!(TextureFormat::RgbDxt1.frame_size(5, 5), 2 * 2 * 8);
    }

    #[test]
    fn single_uncompressed() {
        let payload = vec![0xAA; 32];
        let frame = parse_frame(&section(COMPRESSOR_NONE | FMT_RGB_DXT1, &payload)).unwrap();
        assert_eq!(frame.format, TextureFormat::RgbDxt1);
        assert_eq!(frame.data, payload);
        assert!(frame.alpha.is_none());
    }

    #[test]
    fn single_snappy() {
        let original = vec![0xBB; 64];
        let compressed = snap::raw::Encoder::new().compress_vec(&original).unwrap();
        let frame = parse_frame(&section(COMPRESSOR_SNAPPY | FMT_RGBA_DXT5, &compressed)).unwrap();
        assert_eq!(frame.format, TextureFormat::RgbaDxt5);
        assert_eq!(frame.data, original);
    }

    #[test]
    fn complex_chunked_snappy() {
        // Two Snappy chunks, no offset table (sequential).
        let chunk0 = snap::raw::Encoder::new().compress_vec(&[0x11; 16]).unwrap();
        let chunk1 = snap::raw::Encoder::new().compress_vec(&[0x22; 16]).unwrap();
        let mut frame_data = chunk0.clone();
        frame_data.extend_from_slice(&chunk1);

        let compressors = section(SECTION_CHUNK_COMPRESSORS, &[CHUNK_SNAPPY, CHUNK_SNAPPY]);
        let mut sizes_payload = Vec::new();
        sizes_payload.extend_from_slice(&(chunk0.len() as u32).to_le_bytes());
        sizes_payload.extend_from_slice(&(chunk1.len() as u32).to_le_bytes());
        let sizes = section(SECTION_CHUNK_SIZES, &sizes_payload);

        let mut instructions = compressors;
        instructions.extend_from_slice(&sizes);
        let decode_instr = section(SECTION_DECODE_INSTRUCTIONS, &instructions);

        let mut body = decode_instr;
        body.extend_from_slice(&frame_data);
        let frame = parse_frame(&section(COMPRESSOR_COMPLEX | FMT_RGBA_DXT5, &body)).unwrap();

        let mut expected = vec![0x11; 16];
        expected.extend_from_slice(&[0x22; 16]);
        assert_eq!(frame.data, expected);
    }

    #[test]
    fn multi_image_dual_plane() {
        // Hap Q Alpha: YCoCg color plane + BC4 alpha plane.
        let color = section(COMPRESSOR_NONE | FMT_YCOCG_DXT5, &[0xCC; 16]);
        let alpha = section(COMPRESSOR_NONE | FMT_ALPHA_RGTC1, &[0xDD; 8]);
        let mut payload = color;
        payload.extend_from_slice(&alpha);
        let frame = parse_frame(&section(SECTION_MULTI_IMAGE, &payload)).unwrap();

        assert_eq!(frame.format, TextureFormat::YcoCgDxt5);
        assert_eq!(frame.data, vec![0xCC; 16]);
        let alpha = frame.alpha.expect("dual-plane frame has an alpha plane");
        assert_eq!(alpha.format, TextureFormat::AlphaRgtc1);
        assert_eq!(alpha.data, vec![0xDD; 8]);
    }

    #[test]
    fn detect_format_without_decoding() {
        let single = section(COMPRESSOR_SNAPPY | FMT_RGBA_BC7, &[0xBB; 16]);
        assert_eq!(detect_format(&single).unwrap(), TextureFormat::RgbaBc7);

        let color = section(COMPRESSOR_NONE | FMT_YCOCG_DXT5, &[0xCC; 16]);
        let alpha = section(COMPRESSOR_NONE | FMT_ALPHA_RGTC1, &[0xDD; 8]);
        let mut payload = color;
        payload.extend_from_slice(&alpha);
        let multi = section(SECTION_MULTI_IMAGE, &payload);
        assert_eq!(detect_format(&multi).unwrap(), TextureFormat::YcoCgDxt5);
    }

    #[test]
    fn truncated_frame_errors_not_panics() {
        // A well-formed header claiming a body longer than the buffer.
        let mut bad = section(COMPRESSOR_NONE | FMT_RGB_DXT1, &[0u8; 8]);
        bad.truncate(6); // chop the body
        assert!(parse_frame(&bad).is_err());
    }
}
