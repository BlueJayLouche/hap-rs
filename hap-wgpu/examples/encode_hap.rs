//! Example: Encode HAP video using high-level encoder
//!
//! This example demonstrates how to use the HapVideoEncoder to create
//! HAP videos from generated frame data.
//!
//! Usage:
//!   cargo run --example encode_hap -- [output.mov] [width] [height] [frames]

use hap_wgpu::{EncodeConfig, EncodeQuality, HapFormat, HapVideoEncoder};
use std::path::PathBuf;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    
    let output_path = args.get(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("output_hap.mov")
    });
    
    let width = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let height = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(256);
    let frame_count = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(60);
    
    println!("HAP Video Encoder Example");
    println!("=========================");
    println!("Output: {}", output_path.display());
    println!("Resolution: {}x{}", width, height);
    println!("Frames: {}", frame_count);
    println!();

    // Initialize wgpu
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    
    let adapter = pollster::block_on(async {
        instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("Failed to find suitable adapter")
    });
    
    println!("Using adapter: {:?}", adapter.get_info().name);
    
    // Check BC compression support
    if !adapter.features().contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
        eprintln!("Warning: BC texture compression not supported on this GPU");
        eprintln!("HAP encoding will still work but the resulting files may not be playable");
    }
    
    // Create device and queue
    let (device, queue) = pollster::block_on(async {
        adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::TEXTURE_COMPRESSION_BC,
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                    trace: wgpu::Trace::Off,
                },
            )
            .await
            .expect("Failed to create device")
    });
    
    // Create encoder
    let encoder = HapVideoEncoder::new(Arc::new(device), Arc::new(queue));
    
    // Configure encoding
    let config = EncodeConfig::new(width, height, 30.0, frame_count)
        .with_format(HapFormat::HapY)  // High quality RGB
        .with_quality(EncodeQuality::Balanced)
        .with_snappy(true);
    
    println!("Encoding {} frames...", frame_count);
    let start = std::time::Instant::now();
    
    // Encode video using the frame provider callback
    encoder.encode(&output_path, config, |frame_idx| {
        // Generate test pattern
        let progress = frame_idx as f32 / frame_count as f32;
        let rgba_data = generate_test_pattern(width, height, progress);
        Ok::<Vec<u8>, Box<dyn std::error::Error>>(rgba_data)
    })?;
    
    let duration = start.elapsed();
    println!();
    println!("Encoding complete!");
    println!("Time: {:?}", duration);
    println!("FPS: {:.2}", frame_count as f64 / duration.as_secs_f64());
    println!("Output: {}", output_path.display());
    
    // Print file size
    let metadata = std::fs::metadata(&output_path)?;
    println!("File size: {:.2} MB", metadata.len() as f64 / (1024.0 * 1024.0));
    
    Ok(())
}

/// Generate an animated test pattern
fn generate_test_pattern(width: u32, height: u32, progress: f32) -> Vec<u8> {
    let mut pixels = vec![0u8; (width * height * 4) as usize];
    
    // Create a rotating color wheel pattern
    let center_x = width as f32 / 2.0;
    let center_y = height as f32 / 2.0;
    let max_radius = (width.min(height) as f32) / 2.0;
    
    for y in 0..height {
        for x in 0..width {
            let dx = x as f32 - center_x;
            let dy = y as f32 - center_y;
            let distance = (dx * dx + dy * dy).sqrt();
            
            let angle = dy.atan2(dx);
            let normalized_angle = (angle / std::f32::consts::PI + 1.0) / 2.0;
            
            // Animated rotation
            let hue = (normalized_angle + progress) % 1.0;
            
            // Convert HSV to RGB
            let (r, g, b) = hsv_to_rgb(hue, 1.0, (1.0 - distance / max_radius).max(0.0));
            
            let idx = ((y * width + x) * 4) as usize;
            pixels[idx] = r;
            pixels[idx + 1] = g;
            pixels[idx + 2] = b;
            pixels[idx + 3] = 255;
        }
    }
    
    pixels
}

/// Convert HSV color to RGB
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let h = h * 6.0;
    let i = h.floor() as i32;
    let f = h - i as f32;
    
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    
    let (r, g, b) = match i % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    
    (
        (r * 255.0) as u8,
        (g * 255.0) as u8,
        (b * 255.0) as u8,
    )
}
