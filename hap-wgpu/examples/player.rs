//! Simple HAP Player Example
//! 
//! Demonstrates loading a HAP file and playing it with wgpu.
//! 
//! Note: This example requires a GPU that supports BC texture compression.
//! 
//! Usage:
//!   cargo run --example player --release -- samples/output_converted.hap.mov

use hap_wgpu::{HapPlayer, LoopMode, required_features};
use std::sync::Arc;
use wgpu;
use winit::event::{Event, KeyEvent, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};

fn main() -> anyhow::Result<()> {
    env_logger::init();
    
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <hap-file>", args[0]);
        std::process::exit(1);
    }
    let hap_path = &args[1];
    
    // Create event loop and window
    let event_loop = EventLoop::new()?;
    let window = Arc::new(
        event_loop.create_window(
            winit::window::WindowAttributes::default()
                .with_title("HAP Player Example")
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
        )?
    );
    
    // Create wgpu instance
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let surface = instance.create_surface(window.clone())?;
    
    let adapter = pollster::block_on(async {
        instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }).await.expect("Failed to find suitable adapter")
    });
    
    // Check for BC compression support
    if !adapter.features().contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
        eprintln!("ERROR: This GPU does not support BC texture compression.");
        eprintln!("HAP playback requires the TEXTURE_COMPRESSION_BC feature.");
        std::process::exit(1);
    }
    
    let (device, queue) = pollster::block_on(async {
        adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Device"),
                required_features: required_features(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            },
        ).await.expect("Failed to create device")
    });
    
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    
    // Create HAP player
    println!("Loading HAP file: {}", hap_path);
    let mut player = HapPlayer::open(hap_path, device.clone(), queue.clone())?;
    
    let dimensions = player.dimensions();
    let frame_count = player.frame_count();
    let fps = player.fps();
    let codec = player.codec_type();
    
    println!("Video loaded:");
    println!("  Resolution: {}x{}", dimensions.0, dimensions.1);
    println!("  Frame count: {}", frame_count);
    println!("  FPS: {:.2}", fps);
    println!("  Duration: {:.2}s", player.duration());
    println!("  Codec: {}", codec);
    
    // Set loop mode and start playing
    player.set_loop_mode(LoopMode::Loop);
    player.play();
    
    println!("\nControls:");
    println!("  Space: Play/Pause");
    println!("  R: Restart");
    println!("  Left/Right: Previous/Next frame (when paused)");
    println!("  ESC: Quit");
    
    let mut is_playing = true;
    let mut current_frame = 0u32;
    
    // Main loop
    event_loop.run(move |event, target| {
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => target.exit(),
                
                WindowEvent::KeyboardInput { event, .. } => {
                    if let KeyEvent {
                        physical_key: PhysicalKey::Code(keycode),
                        state: winit::event::ElementState::Pressed,
                        ..
                    } = event {
                        match keycode {
                            KeyCode::Escape => target.exit(),
                            KeyCode::Space => {
                                if is_playing {
                                    player.pause();
                                    println!("Paused at frame {}", current_frame);
                                } else {
                                    player.play();
                                    println!("Playing");
                                }
                                is_playing = !is_playing;
                            }
                            KeyCode::KeyR => {
                                player.seek_to_frame(0);
                                player.play();
                                is_playing = true;
                                println!("Restarted");
                            }
                            KeyCode::ArrowLeft => {
                                if !is_playing && current_frame > 0 {
                                    current_frame -= 1;
                                    player.seek_to_frame(current_frame);
                                    println!("Frame {}", current_frame);
                                }
                            }
                            KeyCode::ArrowRight => {
                                if !is_playing && current_frame < frame_count - 1 {
                                    current_frame += 1;
                                    player.seek_to_frame(current_frame);
                                    println!("Frame {}", current_frame);
                                }
                            }
                            _ => {}
                        }
                    }
                }
                
                WindowEvent::RedrawRequested => {
                    // Update player and get current frame texture
                    if let Some(frame_texture) = player.update() {
                        current_frame = frame_texture.frame_index;
                    }
                    
                    window.request_redraw();
                }
                
                _ => {}
            },
            _ => {}
        }
    })?;
    
    Ok(())
}
