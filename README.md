# HAP

Rust crates for working with HAP video codec.

## Overview

This workspace contains crates for parsing and playing back HAP video files in Rust applications.

## Crates

| Crate | Description |
|-------|-------------|
| [`hap-parser`](./hap-parser/) | HAP video frame parser - decodes HAP frames from raw data |
| [`hap-qt`](./hap-qt/) | QuickTime container reader - extracts HAP frames from .mov files |
| [`hap-wgpu`](./hap-wgpu/) | wgpu integration - renders HAP video to GPU textures |

## Quick Start

```bash
# Clone the repository
git clone https://github.com/BlueJayLouche/hap-rs.git
cd hap-rs

# Build the entire workspace
cargo build --workspace

# Run examples
cargo run --package hap-qt --example hap_info -- path/to/video.mov
cargo run --package hap-parser --example frame_info
cargo run --package hap-wgpu --example player -- path/to/video.mov
```

## HAP Video Format

HAP is a GPU-accelerated video codec designed for high-performance video playback in real-time graphics applications. It uses S3TC/DXT texture compression for efficient GPU upload.

- [HAP Documentation](https://hap.video/)
- Supports HAP, HAP Alpha, HAP Q, HAP Q Alpha variants
- Snappy compression for reduced file sizes

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](./LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](./LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you shall be dual licensed as above, without any additional terms or conditions.
