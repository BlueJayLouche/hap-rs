# Native HAP Encoding Implementation

**Status: ✅ COMPLETE**

Native HAP encoding has been implemented in the `hap-rs` crate. This document describes the implementation for reference.

## Overview

The implementation adds native HAP **encoding** capability to the `hap-rs` crate, allowing applications to create HAP videos without requiring FFmpeg to be installed on end-user machines.

## Architecture

```
hap-rs/
├── hap-parser/     # HAP frame parsing (existing)
├── hap-qt/         # QuickTime container + Writer/Encoder
└── hap-wgpu/       # GPU playback + Video Encoder API
```

## Implementation

### 1. HAP Frame Encoding (`hap-qt/src/frame_encoder.rs`)

Low-level HAP frame encoding using `texpresso` for CPU-based DXT compression:

```rust
pub struct HapFrameEncoder {
    format: HapFormat,
    dimensions: (u32, u32),
}

impl HapFrameEncoder {
    pub fn new(format: HapFormat, width: u32, height: u32) -> Self;
    pub fn encode(&self, rgba_data: &[u8]) -> Result<Vec<u8>>;
    pub fn set_compression(&mut self, mode: CompressionMode);
}
```

**Key Features:**
- Pure Rust implementation using `texpresso` crate
- Supports Hap1 (DXT1), Hap5 (DXT5), HapY (YCoCg-DXT5), HapA (BC4)
- Snappy compression via `snap` crate
- Proper HAP frame header format per specification

### 2. QuickTime Writer (`hap-qt/src/writer.rs`)

QuickTime container writing with proper atom structure:

```rust
pub struct QtHapWriter;

impl QtHapWriter {
    pub fn create(path: &Path, config: VideoConfig) -> Result<Self>;
    pub fn write_frame(&mut self, hap_frame: &[u8]) -> Result<()>;
    pub fn finalize(self) -> Result<()>;
}
```

**Features:**
- Custom QuickTime atom writer (no external mp4 crate dependency)
- All required atoms: ftyp, moov (mvhd, trak, tkhd, mdia, mdhd, minf, stbl, stsd, stts, stsc, stsz, stco), mdat
- Automatic finalization via Drop trait
- Proper chunk offset calculation

### 3. High-Level Encoder (`hap-wgpu/src/encoder.rs`)

Convenient API for encoding complete videos:

```rust
pub struct HapVideoEncoder;

impl HapVideoEncoder {
    pub fn encode<F>(
        &self,
        output_path: &Path,
        config: EncodeConfig,
        frame_provider: F,
    ) -> Result<()>
    where F: FnMut(u32) -> Result<Vec<u8>>;
}
```

## HAP Format Compliance

The implementation follows the [HAP specification](https://github.com/Vidvox/hap/blob/master/documentation/HapVideoDRAFT.md):

### Frame Header Format

```
4-byte header: [size: 3 bytes LE][type: 1 byte]
```

Type bytes:
- 0xAB = Hap1 (DXT1) uncompressed
- 0xBB = Hap1 (DXT1) Snappy compressed
- 0xAE = Hap5 (DXT5) uncompressed
- 0xBE = Hap5 (DXT5) Snappy compressed
- 0xAF = HapY (YCoCg-DXT5) uncompressed
- 0xBF = HapY (YCoCg-DXT5) Snappy compressed

### QuickTime Container

Proper atom structure with:
- ftyp: "qt  " file type
- moov: Complete movie metadata
- mdat: HAP frame data
- stsd: HAP-specific sample description

## Usage Examples

### Basic Encoding

```rust
use hap_qt::{HapFormat, HapFrameEncoder, QtHapWriter, VideoConfig, CompressionMode};

let mut encoder = HapFrameEncoder::new(HapFormat::HapY, 1920, 1080)?;
encoder.set_compression(CompressionMode::Snappy);

let config = VideoConfig::new(1920, 1080, 30.0, HapFormat::HapY);
let mut writer = QtHapWriter::create("output.mov", config)?;

for frame_idx in 0..frame_count {
    let rgba_data = get_frame_data(frame_idx);
    let hap_frame = encoder.encode(&rgba_data)?;
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

## Dependencies

- `texpresso` = "2.0" - CPU DXT compression
- `snap` = "1.1" - Snappy compression
- `byteorder` = "1.5" - Byte order handling

## Limitations

- Hap7 (BC7) and HapH (BC6H) encoding not yet implemented (complex compression)
- GPU-based DXT compression not yet implemented (future optimization)
- Chunked encoding for multi-threaded decoding not implemented

## Resources

- [HAP Video Spec](https://hap.video/)
- [HAP GitHub](https://github.com/vidvox/hap)
- [HAP Specification Draft](https://github.com/Vidvox/hap/blob/master/documentation/HapVideoDRAFT.md)
