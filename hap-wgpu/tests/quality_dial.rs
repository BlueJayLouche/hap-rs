//! The quality preset must actually reach every encoder.
//!
//! This guards a specific regression: a quality field that is stored in a
//! config, threaded through an API, shown in a UI, and then never read by the
//! thing doing the compression. That has happened twice in this codebase, and
//! it is invisible without a test, because everything still encodes fine.
//!
//! Proving "Best is better than Fast" needs a decoder per format; proving the
//! knob is *connected* only needs the output to change, which is what this
//! checks. Skips gracefully when no BC-capable adapter is available.

use hap_qt::{CompressionMode, DxtQuality, HapFormat, HapFrameEncoder};
use hap_wgpu::GpuDxtCompressor;
use std::sync::Arc;

const W: u32 = 128;
const H: u32 = 128;

/// Four quadrants: smooth ramp, hard edges, noise, and fine detail.
///
/// The content matters. A checkerboard coarser than 4px makes every block a
/// single colour, which every algorithm encodes identically — such an image
/// "passes" by proving nothing.
fn test_image() -> Vec<u8> {
    let mut v = vec![0u8; (W * H * 4) as usize];
    let mut st = 0x9E37_79B9u32;
    for y in 0..H {
        for x in 0..W {
            st ^= st << 13;
            st ^= st >> 17;
            st ^= st << 5;
            let i = ((y * W + x) * 4) as usize;
            let (r, g, b) = match (x < W / 2, y < H / 2) {
                (true, true) => ((x * 255 / W) as u8, (y * 255 / H) as u8, 128),
                (false, true) => {
                    let on = (x / 2 + y / 2) % 2 == 0;
                    (if on { 240 } else { 10 }, if on { 30 } else { 40 }, if on { 20 } else { 220 })
                }
                (true, false) => (st as u8, (st >> 8) as u8, (st >> 16) as u8),
                (false, false) => {
                    let d = (((x * y) % 97) * 255 / 97) as u8;
                    (d, d.wrapping_add(60), 255 - d)
                }
            };
            v[i] = r;
            v[i + 1] = g;
            v[i + 2] = b;
            v[i + 3] = (st >> 24) as u8;
        }
    }
    v
}

const FORMATS: [HapFormat; 5] = [
    HapFormat::Hap1,
    HapFormat::Hap5,
    HapFormat::HapY,
    HapFormat::HapA,
    HapFormat::Hap7,
];

#[test]
fn cpu_quality_reaches_every_encoder() {
    let img = test_image();
    for fmt in FORMATS {
        let encode = |q| {
            let mut enc = HapFrameEncoder::new(fmt, W, H).unwrap();
            enc.set_compression(CompressionMode::None);
            enc.set_quality(q);
            enc.encode(&img).unwrap()
        };
        let fast = encode(DxtQuality::Fast);
        let best = encode(DxtQuality::Best);

        if fmt == HapFormat::HapA {
            // texpresso compresses BC4 with a single fixed algorithm, so the
            // preset genuinely cannot change this one. Asserting it does would
            // be asserting a lie.
            assert_eq!(fast, best, "{fmt:?}: expected BC4 to be quality-invariant");
        } else {
            assert_ne!(fast, best, "{fmt:?}: CPU quality preset had no effect");
        }
    }
}

#[test]
fn gpu_quality_reaches_every_shader() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
        eprintln!("skipping: no adapter");
        return;
    };
    if !adapter.features().contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
        eprintln!("skipping: no BC support");
        return;
    }
    let Ok((device, queue)) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("quality-dial-test"),
        required_features: wgpu::Features::TEXTURE_COMPRESSION_BC,
        ..Default::default()
    })) else {
        eprintln!("skipping: no device");
        return;
    };
    let Some(gpu) = GpuDxtCompressor::try_new(Arc::new(device), Arc::new(queue), W, H) else {
        eprintln!("skipping: no GPU compressor");
        return;
    };

    let img = test_image();
    for fmt in FORMATS {
        let fast = gpu.compress(&img, fmt, DxtQuality::Fast).unwrap();
        let best = gpu.compress(&img, fmt, DxtQuality::Best).unwrap();
        assert_ne!(fast, best, "{fmt:?}: GPU quality preset had no effect");
    }
}
