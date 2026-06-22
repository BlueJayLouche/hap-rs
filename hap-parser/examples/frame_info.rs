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
            println!("Texture format: {:?}", frame.format);
            println!("Needs YCoCg convert: {}", frame.format.needs_ycocg_convert());
            println!("Texture data size: {} bytes", frame.data.len());
            if let Some(alpha) = &frame.alpha {
                println!("Alpha plane: {:?}, {} bytes", alpha.format, alpha.data.len());
            }

            // Calculate expected size for common resolutions
            let expected_720p = frame.format.frame_size(1280, 720);
            let expected_1080p = frame.format.frame_size(1920, 1080);

            println!("\nExpected sizes:");
            println!("  1280x720: {} bytes", expected_720p);
            println!("  1920x1080: {} bytes", expected_1080p);

            if frame.data.len() == expected_720p {
                println!("  -> Matches 720p");
            } else if frame.data.len() == expected_1080p {
                println!("  -> Matches 1080p");
            } else {
                let blocks = frame.data.len() / frame.format.bytes_per_block();
                println!("  -> {} blocks", blocks);
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
