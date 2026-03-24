//! Example: Encode a simple HAP video
//!
//! This example creates a test HAP video with generated frames.

use hap_qt::{HapFormat, HapFrameEncoder, QtHapWriter, VideoConfig, CompressionMode};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    
    let output_path = args.get(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("test_output.mov")
    });
    
    let width = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let height = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(256);
    let frame_count = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(30);
    
    println!("Creating HAP video:");
    println!("  Output: {}", output_path.display());
    println!("  Resolution: {}x{}", width, height);
    println!("  Frames: {}", frame_count);
    println!("  Format: HapY (YCoCg-DXT5)");
    println!();

    // Create video configuration
    let config = VideoConfig::new(width, height, 30.0, HapFormat::HapY);
    
    // Create frame encoder
    let mut encoder = HapFrameEncoder::new(HapFormat::HapY, width, height)?;
    encoder.set_compression(CompressionMode::Snappy);
    
    // Create video writer
    let mut writer = QtHapWriter::create(&output_path, config)?;
    
    println!("Encoding frames...");
    
    // Generate and encode frames
    for frame_idx in 0..frame_count {
        // Generate a simple test pattern (gradient)
        let rgba_data = generate_gradient_frame(width, height, frame_idx as f32 / frame_count as f32);
        
        // Encode to HAP
        let hap_frame = encoder.encode(&rgba_data)?;
        
        // Write frame
        writer.write_frame(&hap_frame)?;
        
        if frame_idx % 10 == 0 {
            println!("  Encoded frame {}/{}", frame_idx, frame_count);
        }
    }
    
    // Finalize the video file
    writer.finalize()?;
    
    println!();
    println!("Successfully created: {}", output_path.display());
    println!("File size: {} bytes", std::fs::metadata(&output_path)?.len());
    
    Ok(())
}

/// Generate a gradient test pattern
fn generate_gradient_frame(width: u32, height: u32, phase: f32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    
    for y in 0..height {
        for x in 0..width {
            // Create a gradient with animation
            let r = ((x as f32 / width as f32) * 255.0) as u8;
            let g = ((y as f32 / height as f32) * 255.0) as u8;
            let b = ((phase * 255.0) as u8).wrapping_add(128);
            let a = 255u8;
            
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
            pixels.push(a);
        }
    }
    
    pixels
}
