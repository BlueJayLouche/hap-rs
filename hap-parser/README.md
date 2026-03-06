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
  - Hap R Alpha (BC6H + Alpha)
- Snappy decompression
- Decode instruction parsing (for complex frames)

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
[Section Header 4 or 8 bytes]
[Section Data]
```

The section header indicates the size and type of the section. For simple frames, the section data is the compressed texture. For complex frames, it contains decode instructions.

## License

MIT OR Apache-2.0
