# hap-qt

QuickTime container reader and writer for HAP video files.

## Overview

This crate provides:
- **Reading**: Parse QuickTime/MP4 containers to extract HAP video frames
- **Writing**: Create QuickTime containers with HAP video tracks

No FFmpeg required - pure Rust implementation.

## Features

- Read HAP video from .mov files
- Write HAP video to .mov files
- Support for all HAP variants:
  - Hap1 (RGB DXT1)
  - Hap5 (RGBA DXT5)
  - HapY (YCoCg DXT5)
  - HapA (Alpha BC4)
  - Hap7 (BC7) - read only
  - HapH (BC6H HDR) - read only
- Snappy compression/decompression

## Usage

### Reading HAP Video

```rust
use hap_qt::QtHapReader;

let mut reader = QtHapReader::open("video.mov")?;

println!("Resolution: {}x{}", reader.resolution().0, reader.resolution().1);
println!("Frames: {}", reader.frame_count());
println!("FPS: {}", reader.fps());

// Read specific frame
let frame = reader.read_frame(0)?;
println!("Format: {:?}", frame.texture_format);
```

### Writing HAP Video

```rust
use hap_qt::{HapFormat, HapFrameEncoder, QtHapWriter, VideoConfig, CompressionMode};

// Create frame encoder
let mut encoder = HapFrameEncoder::new(HapFormat::HapY, 1920, 1080)?;
encoder.set_compression(CompressionMode::Snappy);

// Create video writer
let config = VideoConfig::new(1920, 1080, 30.0, HapFormat::HapY);
let mut writer = QtHapWriter::create("output.mov", config)?;

// Encode and write frames
for i in 0..300 {
    let rgba_data = generate_frame(i); // Your frame data
    let hap_frame = encoder.encode(&rgba_data)?;
    writer.write_frame(&hap_frame)?;
}

// Finalize (required!)
writer.finalize()?;
```

## QuickTime Container Structure

The reader/writer handles these atoms:

1. **ftyp** - File type identification ("qt  ")
2. **moov** - Movie metadata
   - **trak** - Video track
     - **tkhd** - Track header (dimensions)
     - **mdia** - Media information
       - **mdhd** - Timescale and duration
       - **hdlr** - Handler type ("vide")
       - **minf** - Media info
         - **vmhd** - Video media header
         - **dinf** - Data information
         - **stbl** - Sample table
           - **stsd** - Sample descriptions (codec: Hap1/Hap5/HapY/etc)
           - **stts** - Time-to-sample
           - **stsc** - Sample-to-chunk
           - **stsz** - Sample sizes
           - **stco** - Chunk offsets
3. **mdat** - Media data (HAP frames)

## API Reference

### `QtHapReader`

```rust
impl QtHapReader {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self>;
    pub fn resolution(&self) -> (u32, u32);
    pub fn frame_count(&self) -> u32;
    pub fn fps(&self) -> f32;
    pub fn read_frame(&mut self, index: u32) -> Result<HapFrame>;
}
```

### `HapFrameEncoder`

```rust
impl HapFrameEncoder {
    pub fn new(format: HapFormat, width: u32, height: u32) -> Result<Self>;
    pub fn encode(&self, rgba_data: &[u8]) -> Result<Vec<u8>>;
    pub fn set_compression(&mut self, mode: CompressionMode);
}
```

### `QtHapWriter`

```rust
impl QtHapWriter {
    pub fn create<P: AsRef<Path>>(path: P, config: VideoConfig) -> Result<Self>;
    pub fn write_frame(&mut self, hap_frame: &[u8]) -> Result<()>;
    pub fn finalize(self) -> Result<()>;
    pub fn frame_count(&self) -> u32;
}
```

## HAP Frame Format

HAP frames follow the [HAP specification](https://github.com/Vidvox/hap/blob/master/documentation/HapVideoDRAFT.md):

```
[Header: 4 bytes] [Compressed Data]
```

Header: `[size: 3 bytes LE][type: 1 byte]`

Types:
- 0xAB = Hap1 uncompressed, 0xBB = Hap1 Snappy
- 0xAE = Hap5 uncompressed, 0xBE = Hap5 Snappy
- 0xAF = HapY uncompressed, 0xBF = HapY Snappy

## Examples

```bash
# Run encode example
cargo run --example encode_video -- output.mov 256 256 60
```

See `examples/encode_video.rs` for a complete example.

## License

MIT OR Apache-2.0
