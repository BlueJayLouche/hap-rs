//! HAP Video File Info
//!
//! Usage: cargo run --example hap_info -- <hap_mov_file>

use std::env;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        eprintln!("Usage: {} <hap_mov_file>", args[0]);
        std::process::exit(1);
    }
    
    let path = &args[1];
    println!("Opening: {}", path);
    println!();
    
    let reader = hap_qt::QtHapReader::open(path)?;
    
    println!("=== Video Info ===");
    println!("Resolution: {}x{}", reader.resolution().0, reader.resolution().1);
    println!("Frame count: {}", reader.frame_count());
    println!("Frame rate: {:.2} fps", reader.fps());
    println!("Duration: {:.2} seconds", reader.duration());
    println!("Codec: {}", reader.codec_type());
    
    // Try to read first frame
    println!("\n=== First Frame ===");
    let mut reader = reader; // Re-bind as mutable
    match reader.read_frame(0) {
        Ok(frame) => {
            println!("Format: {:?}", frame.texture_format);
            println!("Uses Snappy: {}", frame.uses_snappy);
            println!("Texture data size: {} bytes", frame.texture_data.len());
            
            // Calculate expected size
            let expected = hap_parser::expected_texture_size(
                frame.texture_format,
                reader.resolution().0,
                reader.resolution().1
            );
            println!("Expected size for {}x{}: {} bytes", 
                reader.resolution().0,
                reader.resolution().1,
                expected
            );
            
            if frame.texture_data.len() == expected {
                println!("✓ Size matches expected");
            } else {
                println!("⚠ Size mismatch! Got {} bytes", frame.texture_data.len());
            }
        }
        Err(e) => {
            eprintln!("Failed to read first frame: {}", e);
        }
    }
    
    // Try to read last frame
    println!("\n=== Last Frame ===");
    let last_frame = reader.frame_count() - 1;
    match reader.read_frame(last_frame) {
        Ok(frame) => {
            println!("Successfully read frame {}", last_frame);
            println!("Format: {:?}", frame.texture_format);
            println!("Texture data size: {} bytes", frame.texture_data.len());
        }
        Err(e) => {
            eprintln!("Failed to read last frame: {}", e);
        }
    }
    
    Ok(())
}
