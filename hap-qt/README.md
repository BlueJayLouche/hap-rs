# hap-qt

QuickTime container reader for HAP video files.

## Overview

This crate parses QuickTime/MP4 containers to extract HAP video frames without requiring ffmpeg. It reads the moov atom (movie metadata) and mdat atom (media data) directly.

## Usage

```rust
use hap_qt::QtHapReader;

let reader = QtHapReader::open("video.mov")?;

println!("Resolution: {}x{}", reader.resolution().0, reader.resolution().1);
println!("Frames: {}", reader.frame_count());
println!("FPS: {}", reader.fps());

// Read specific frame
let mut reader = reader; // Make mutable
let frame = reader.read_frame(0)?;

println!("Format: {:?}", frame.texture_format);
println!("Compressed size: {} bytes", frame.texture_data.len());
```

## Supported Codecs

- Hap1 (RGB DXT1)
- Hap5 (RGBA DXT5)
- HapY (YCoCg DXT5)
- HapM (YCoCg + Alpha)
- HapA (Alpha only)
- Hap7 (BC7)
- HapH (BC6H HDR)

## Architecture

The reader parses:
1. **ftyp** - File type identification
2. **moov** - Movie metadata
   - **trak** - Video track
     - **tkhd** - Track header (dimensions)
     - **mdia** - Media information
       - **mdhd** - Timescale and duration
       - **minf** - Media info
         - **stbl** - Sample table
           - **stsd** - Sample descriptions (codec)
           - **stsz** - Sample sizes
           - **stco** - Chunk offsets
           - **stsc** - Sample-to-chunk mapping
3. **mdat** - Media data (referenced by offsets)

## License

MIT OR Apache-2.0
