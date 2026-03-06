//! HAP Frame Inspector
//!
//! Usage: cargo run --example frame_info -- <hap_file>

use std::env;
use std::fs;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        eprintln!("Usage: {} <hap_file>", args[0]);
        eprintln!("");
        eprintln!("Inspects the first HAP frame in a QuickTime container.");
        std::process::exit(1);
    }
    
    let path = &args[1];
    println!("Reading: {}", path);
    
    // Read file
    let data = fs::read(path)?;
    println!("File size: {} bytes", data.len());
    
    // Try to find and parse HAP frames
    // For now, just try parsing from offset 0 (raw HAP file)
    // In a real implementation, we'd parse the QuickTime container
    
    match hap_parser::parse_frame(&data) {
        Ok(frame) => {
            println!("\n=== HAP Frame Info ===");
            println!("Top-level type: {:?}", frame.top_level_type);
            println!("Texture format: {:?}", frame.texture_format);
            println!("Uses Snappy: {}", frame.uses_snappy);
            println!("Texture data size: {} bytes", frame.texture_data.len());
            
            if !frame.chunks.is_empty() {
                println!("Chunks: {}", frame.chunks.len());
                for (i, chunk) in frame.chunks.iter().enumerate() {
                    println!("  Chunk {}: {:?}, size={}", i, chunk.compressor, chunk.size);
                }
            }
            
            // Calculate expected size for common resolutions
            let expected_720p = hap_parser::expected_texture_size(frame.texture_format, 1280, 720);
            let expected_1080p = hap_parser::expected_texture_size(frame.texture_format, 1920, 1080);
            
            println!("\nExpected sizes:");
            println!("  1280x720: {} bytes", expected_720p);
            println!("  1920x1080: {} bytes", expected_1080p);
            
            if frame.texture_data.len() == expected_720p {
                println!("  -> Matches 720p");
            } else if frame.texture_data.len() == expected_1080p {
                println!("  -> Matches 1080p");
            } else {
                // Try to guess resolution
                let block_size = hap_parser::bytes_per_block(frame.texture_format);
                let blocks = frame.texture_data.len() / block_size;
                println!("  -> {} blocks ({}x{} approx)", blocks, 
                    (blocks as f32).sqrt() as u32 * 4,
                    (blocks as f32).sqrt() as u32 * 4);
            }
        }
        Err(e) => {
            eprintln!("Failed to parse HAP frame: {}", e);
            eprintln!("\nNote: This tool expects raw HAP frame data.");
            eprintln!("For QuickTime containers, use hap-qt crate (coming soon).");
        }
    }
    
    Ok(())
}
