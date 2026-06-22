# hap-parser

Parse [HAP](https://hap.video) video frames in pure Rust — no FFmpeg.

HAP is a GPU-accelerated codec: each frame is a block-compressed (BCn) texture
that uploads straight to the GPU with no CPU pixel decode. This crate turns one
raw HAP frame (a packet from a QuickTime/MP4 container) into ready-to-upload
texture data.

## Supported formats

| Variant       | Texture format    | Notes                                     |
|---------------|-------------------|-------------------------------------------|
| Hap           | RGB DXT1 (BC1)    |                                           |
| Hap Alpha     | RGBA DXT5 (BC3)   |                                           |
| Hap Q         | scaled YCoCg DXT5 | needs YCoCg→RGB conversion before display |
| Hap Q Alpha   | YCoCg + BC4 alpha | dual-plane (`HapFrame::alpha`)            |
| Hap R         | RGBA BC7          |                                           |

Each variant may be stored uncompressed, Snappy-compressed, or chunked
(complex) — all three are handled transparently.

## Usage

```rust
let frame = hap_parser::parse_frame(frame_bytes)?;

println!("{:?}, {} bytes", frame.format, frame.data.len());
if frame.format.needs_ycocg_convert() {
    // Hap Q: convert YCoCg→RGB in a shader after GPU decode.
}
if let Some(alpha) = &frame.alpha {
    // Hap Q Alpha: separate BC4 alpha plane.
}
```

To size GPU textures before decoding (e.g. when probing the first frame of an
ffmpeg stream, which reports every HAP variant under one codec id), read just
the format:

```rust
let format = hap_parser::detect_format(frame_bytes)?;
```

## Frame layout

Each section has a 4-byte header `[len: u24 LE][type: u8]`, or an 8-byte header
`[0,0,0][type][len: u32 LE]` when the 3-byte length is zero. A top-level
section's type byte encodes the texture format in its low nibble and the
compressor in its high nibble (`0xA` none, `0xB` Snappy, `0xC` complex). Section
type `0x0D` is a multi-image container (Hap Q Alpha).

The parser bounds-checks every section against the buffer, so malformed input
returns a `HapError` rather than panicking.

## License

MIT OR Apache-2.0
