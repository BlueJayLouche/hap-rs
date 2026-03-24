# hap-parser

A Rust crate for parsing HAP video frames.

HAP is a GPU-accelerated video codec designed for real-time playback of high-resolution video. It stores frames as compressed textures (DXT/BCn formats) that can be uploaded directly to the GPU without CPU decompression.

## Features

- Parse HAP frame headers and sections
- Support for all HAP variants:
  - Hap (RGB DXT1)
  - Hap Alpha (RGBA DXT5)
  - Hap Q (YCoCg DXT5)
  - Hap Q Alpha (YCoCg + Alpha)
  - Hap R (BC6H HDR)
  - Hap R Alpha (BC7)
- Snappy decompression
- Decode instruction parsing (for complex frames with multiple chunks)

## Usage

```rust
use hap_parser::parse_frame;

// Parse a HAP frame from raw bytes
let frame_data: &[u8] = // ... read from file
let frame = parse_frame(frame_data)?;

println!("Format: {:?}", frame.texture_format);
println!("Compressed size: {} bytes", frame.texture_data.len());
```

## HAP Frame Structure

HAP frames use a section-based layout:

```
[Section Header: 4 or 8 bytes]
[Section Data]
```

### Section Header

- **4-byte header**: `[size: 3 bytes LE][type: 1 byte]` when size < 16MB
- **8-byte header**: `[0,0,0][type][size: 4 bytes LE]` when size >= 16MB

### Top-Level Section Types

| Type | Format | Compression |
|------|--------|-------------|
| 0xAB | RGB DXT1 | None |
| 0xBB | RGB DXT1 | Snappy |
| 0xAE | RGBA DXT5 | None |
| 0xBE | RGBA DXT5 | Snappy |
| 0xAF | YCoCg DXT5 | None |
| 0xBF | YCoCg DXT5 | Snappy |
| 0xA1 | Alpha BC4 | None |
| 0xB1 | Alpha BC4 | Snappy |

For compressed frames, the section data is Snappy-compressed. After decompression, the result is raw DXT/BCn texture data.

## License

MIT OR Apache-2.0
