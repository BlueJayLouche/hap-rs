//! Time the CPU BC7 encoder at each quality preset.

use hap_qt::DxtQuality;

fn main() {
    let w = 1920usize;
    let h = 1080usize;
    let mut rgba = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            rgba[i] = (x % 256) as u8;
            rgba[i + 1] = (y % 256) as u8;
            rgba[i + 2] = ((x + y) % 256) as u8;
            rgba[i + 3] = 255;
        }
    }
    let mut out = vec![0u8; w * h];
    // One warm-up pass, then take the best of several runs: a single timing is
    // dominated by cache warming and reorders the presets.
    hap_qt::bc7::compress_bc7_mode6(&rgba, w, h, 0, &mut out);
    for q in [DxtQuality::Fast, DxtQuality::Balanced, DxtQuality::Best] {
        let mut best = std::time::Duration::MAX;
        for _ in 0..5 {
            let t = std::time::Instant::now();
            hap_qt::bc7::compress_bc7_mode6(&rgba, w, h, q.refine_iters(), &mut out);
            best = best.min(t.elapsed());
        }
        println!(
            "1080p BC7 mode6, {:<8} ({} refits): {:?}",
            format!("{q:?}"),
            q.refine_iters(),
            best
        );
    }
}
