//! Headless playback check: drives `HapPlayer::update()` at a fixed call rate
//! and reports what read-ahead costs and delivers.
//!
//! ```sh
//! cargo run --release -p hap-wgpu --example playback_rate -- clip.mov [seconds] [calls_per_sec]
//! ```
//!
//! In steady state a healthy player shows ~the clip's fps in distinct frames
//! per second and decodes about as many. Decodes well above frames shown means
//! read-ahead is fetching frames nobody displays; frames shown well below the
//! clip rate means the decoder is not keeping up (or the playhead is slow).

use std::sync::Arc;
use std::time::{Duration, Instant};

use hap_wgpu::{HapPlayer, LoopMode};

/// Process CPU seconds, from `ps` (no extra dependency for a dev tool).
fn cpu_seconds() -> f64 {
    let out = std::process::Command::new("ps")
        .args(["-o", "time=", "-p", &std::process::id().to_string()])
        .output()
        .expect("ps");
    // [[dd-]hh:]mm:ss[.ff]
    let s = String::from_utf8_lossy(&out.stdout);
    s.trim()
        .split(['-', ':'])
        .fold(0.0, |acc, part| acc * 60.0 + part.parse::<f64>().unwrap_or(0.0))
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: playback_rate <clip.mov> [seconds] [calls_per_sec]");
    let seconds: f64 = args.next().map_or(10.0, |s| s.parse().expect("seconds"));
    let rate: f64 = args.next().map_or(60.0, |s| s.parse().expect("calls_per_sec"));

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))?;
    anyhow::ensure!(hap_wgpu::is_supported(&adapter), "adapter lacks BC texture compression");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("playback_rate"),
        required_features: hap_wgpu::required_features(),
        ..Default::default()
    }))?;
    let (device, queue) = (Arc::new(device), Arc::new(queue));

    let mut player = HapPlayer::open(&path, device.clone(), queue.clone())?;
    player.set_loop_mode(LoopMode::Loop);
    player.play();

    // Let the read-ahead fill before measuring.
    let warmup_until = Instant::now() + Duration::from_millis(500);
    let tick = Duration::from_secs_f64(1.0 / rate);
    let mut next = Instant::now();
    while Instant::now() < warmup_until {
        player.update();
        next += tick;
        std::thread::sleep(next.saturating_duration_since(Instant::now()));
    }

    let (cpu0, decodes0, t0) = (cpu_seconds(), player.decode_count(), Instant::now());
    let (mut calls, mut shown, mut misses, mut last) = (0u64, 0u64, 0u64, None);
    let end = t0 + Duration::from_secs_f64(seconds);
    while Instant::now() < end {
        match player.update() {
            Some(tex) if Some(tex.frame_index) != last => {
                shown += 1;
                last = Some(tex.frame_index);
            }
            Some(_) => {}
            None => misses += 1,
        }
        let _ = device.poll(wgpu::PollType::Poll);
        calls += 1;
        next += tick;
        std::thread::sleep(next.saturating_duration_since(Instant::now()));
    }
    let wall = t0.elapsed().as_secs_f64();
    let cpu = cpu_seconds() - cpu0;
    let decodes = player.decode_count() - decodes0;

    println!(
        "clip {:.2} fps | calls {:.1}/s | frames shown {:.1}/s | decodes {:.1}/s ({:.2} per shown) | misses {} | CPU {:.1}%",
        player.fps(),
        calls as f64 / wall,
        shown as f64 / wall,
        decodes as f64 / wall,
        decodes as f64 / shown.max(1) as f64,
        misses,
        100.0 * cpu / wall,
    );
    Ok(())
}
