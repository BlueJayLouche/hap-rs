# HAP

Rust crates for working with HAP video codec - pure Rust implementation with no FFmpeg dependency for encoding.

## Overview

This workspace provides tools for encoding, decoding, and playing back HAP video files in Rust applications:

- **Encode** videos to HAP format natively (no FFmpeg required)
- **Decode** HAP frames from QuickTime containers
- **Playback** HAP video with GPU acceleration via wgpu

## Crates

| Crate | Description |
|-------|-------------|
| [`hap-parser`](./hap-parser/) | Low-level HAP frame parser |
| [`hap-qt`](./hap-qt/) | QuickTime container reader/writer |
| [`hap-wgpu`](./hap-wgpu/) | wgpu integration for GPU playback |

## Features

### Encoding
- ✅ Native HAP encoding without FFmpeg
- ✅ Formats: Hap1 (DXT1), Hap5 (DXT5), HapY (YCoCg-DXT5), HapA (BC4)
- ✅ Snappy compression
- ✅ CPU-based DXT compression via `texpresso`

### Decoding
- ✅ Parse HAP frames from QuickTime .mov files
- ✅ GPU-accelerated playback
- ✅ All HAP variants supported

## Quick Start

```bash
# Clone and build
git clone https://github.com/BlueJayLouche/hap-rs.git
cd hap-rs
cargo build --workspace

# Run examples
cargo run --package hap-qt --example encode_video -- output.mov 256 256 60
cargo run --package hap-wgpu --example encode_hap -- output.mov 512 512 120
cargo run --package hap-qt --example player -- video.mov
```

## Usage

### Encoding

```rust
use hap_qt::{HapFormat, HapFrameEncoder, QtHapWriter, VideoConfig, CompressionMode};

// Create encoder
let mut encoder = HapFrameEncoder::new(HapFormat::HapY, 1920, 1080)?;
encoder.set_compression(CompressionMode::Snappy);

// Create writer
let config = VideoConfig::new(1920, 1080, 30.0, HapFormat::HapY);
let mut writer = QtHapWriter::create("output.mov", config)?;

// Encode frames
for i in 0..300 {
    let rgba = get_frame_data(i);
    let hap_frame = encoder.encode(&rgba)?;
    writer.write_frame(&hap_frame)?;
}

writer.finalize()?;
```

### High-Level API

```rust
use hap_wgpu::{HapVideoEncoder, EncodeConfig, HapFormat};

let encoder = HapVideoEncoder::new(device, queue);
let config = EncodeConfig::new(1920, 1080, 30.0, 300)
    .with_format(HapFormat::HapY);

encoder.encode("output.mov", config, |frame_idx| {
    Ok(get_frame_pixels(frame_idx))
})?;
```

### Decoding

```rust
use hap_qt::QtHapReader;

let mut reader = QtHapReader::open("video.mov")?;
println!("{}x{} @ {} fps", reader.resolution().0, reader.resolution().1, reader.fps());

let frame = reader.read_frame(0)?;
println!("Format: {:?}", frame.texture_format);
```

## HAP Video Format

HAP is a GPU-accelerated video codec using S3TC/DXT texture compression:

- [HAP Website](https://hap.video/)
- [HAP Specification](https://github.com/Vidvox/hap/blob/master/documentation/HapVideoDRAFT.md)
- Optimized for real-time playback in VJ software

### Supported Formats

| Format | Compression | Use Case |
|--------|-------------|----------|
| Hap1 | DXT1 | RGB, smallest size |
| Hap5 | DXT5 | RGBA with alpha |
| HapY | YCoCg-DXT5 | High quality RGB |
| HapA | BC4 | Alpha only |
| Hap7 | BC7 | High quality RGBA (decode only) |
| HapH | BC6H | HDR (decode only) |

## License

MIT OR Apache-2.0

## Contributing

Contributions welcome! Please ensure your code follows the existing style and includes tests.
