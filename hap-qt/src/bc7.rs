//! Minimal BC7 encoder using mode 6 only.
//!
//! Mode 6: single subset, RGBA 7-bit endpoints + 1 shared p-bit per endpoint,
//! 4-bit interpolation indices. One 16-byte block per 4x4 pixels.
//!
//! The encoder picks endpoints with PCA (power iteration over the 4D colour
//! covariance), quantizes to 7 bits + p-bit, and assigns indices by nearest
//! palette entry. `refine_iters` then runs least-squares endpoint refits
//! against the fixed indices, keeping each result only if it lowers the
//! block's squared error. If pixel 0 lands on an index >= 8 the endpoints are
//! swapped (indices map `i -> 15 - i` exactly, since the 4-bit weight table
//! satisfies `w[15 - i] == 64 - w[i]`), because the anchor index is stored in
//! 3 bits.

/// 4-bit interpolation weights for BC7 modes 3/6/7 (out of 64).
const WEIGHTS4: [u32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

/// Compress a padded RGBA image (width/height multiples of 4) to BC7 mode 6.
///
/// `out` must be `width * height` bytes (16 bytes per 4x4 block).
/// `refine_iters` is the maximum number of least-squares endpoint refits per
/// block; 0 is the plain PCA fit.
pub fn compress_bc7_mode6(
    rgba: &[u8],
    width: usize,
    height: usize,
    refine_iters: u32,
    out: &mut [u8],
) {
    debug_assert_eq!(width % 4, 0);
    debug_assert_eq!(height % 4, 0);
    debug_assert_eq!(rgba.len(), width * height * 4);
    debug_assert_eq!(out.len(), width * height);

    for by in 0..height / 4 {
        for bx in 0..width / 4 {
            let mut px = [[0u8; 4]; 16];
            for (i, p) in px.iter_mut().enumerate() {
                let x = bx * 4 + i % 4;
                let y = by * 4 + i / 4;
                p.copy_from_slice(&rgba[(y * width + x) * 4..(y * width + x) * 4 + 4]);
            }
            let block = encode_block(&px, refine_iters);
            out[(by * width / 4 + bx) * 16..(by * width / 4 + bx) * 16 + 16]
                .copy_from_slice(&block);
        }
    }
}

fn encode_block(px: &[[u8; 4]; 16], refine_iters: u32) -> [u8; 16] {
    // Mean and 4x4 covariance in RGBA space.
    let mut mean = [0f32; 4];
    for p in px {
        for c in 0..4 {
            mean[c] += p[c] as f32;
        }
    }
    for c in 0..4 {
        mean[c] /= 16.0;
    }
    let mut cov = [[0f32; 4]; 4];
    for p in px {
        let d: [f32; 4] = [
            p[0] as f32 - mean[0],
            p[1] as f32 - mean[1],
            p[2] as f32 - mean[2],
            p[3] as f32 - mean[3],
        ];
        for i in 0..4 {
            for j in 0..4 {
                cov[i][j] += d[i] * d[j];
            }
        }
    }

    // Principal axis via power iteration, seeded with the max-range channel.
    let mut range = [0f32; 4];
    for c in 0..4 {
        let mut lo = 255f32;
        let mut hi = 0f32;
        for p in px {
            lo = lo.min(p[c] as f32);
            hi = hi.max(p[c] as f32);
        }
        range[c] = hi - lo;
    }
    let seed = range
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let mut axis = [0f32; 4];
    axis[seed] = 1.0;
    for _ in 0..10 {
        let mut v = [0f32; 4];
        for i in 0..4 {
            for j in 0..4 {
                v[i] += cov[i][j] * axis[j];
            }
        }
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt();
        if n < 1e-6 {
            break; // flat block; axis irrelevant
        }
        axis = [v[0] / n, v[1] / n, v[2] / n, v[3] / n];
    }

    // Endpoints at the extremes of the projection onto the axis.
    let mut t_min = f32::MAX;
    let mut t_max = f32::MIN;
    for p in px {
        let t: f32 = (0..4).map(|c| (p[c] as f32 - mean[c]) * axis[c]).sum();
        t_min = t_min.min(t);
        t_max = t_max.max(t);
    }
    let e0: [f32; 4] = [
        (mean[0] + t_min * axis[0]).clamp(0.0, 255.0),
        (mean[1] + t_min * axis[1]).clamp(0.0, 255.0),
        (mean[2] + t_min * axis[2]).clamp(0.0, 255.0),
        (mean[3] + t_min * axis[3]).clamp(0.0, 255.0),
    ];
    let e1: [f32; 4] = [
        (mean[0] + t_max * axis[0]).clamp(0.0, 255.0),
        (mean[1] + t_max * axis[1]).clamp(0.0, 255.0),
        (mean[2] + t_max * axis[2]).clamp(0.0, 255.0),
        (mean[3] + t_max * axis[3]).clamp(0.0, 255.0),
    ];

    let (mut q0, mut p0) = quantize_endpoint(&e0);
    let (mut q1, mut p1) = quantize_endpoint(&e1);

    // Index of each pixel against the quantized palette line.
    let mut idx = assign_indices(px, &q0, p0, &q1, p1);

    // Least-squares refinement: refit the endpoints to the indices we just
    // assigned, then re-index. Each round is kept only if it lowers the error,
    // so more iterations can never make a block worse.
    let mut best_err = block_error(px, &q0, p0, &q1, p1, &idx);
    for _ in 0..refine_iters {
        let Some((r0, r1)) = refit_endpoints(px, &idx) else {
            break; // degenerate: every pixel on one index
        };
        let (nq0, np0) = quantize_endpoint(&r0);
        let (nq1, np1) = quantize_endpoint(&r1);
        let nidx = assign_indices(px, &nq0, np0, &nq1, np1);
        let err = block_error(px, &nq0, np0, &nq1, np1, &nidx);
        if err >= best_err {
            break; // converged
        }
        best_err = err;
        (q0, p0, q1, p1, idx) = (nq0, np0, nq1, np1, nidx);
    }

    // The anchor index (pixel 0) is stored in 3 bits, so it must be < 8.
    // Swapping endpoints remaps indices exactly via i -> 15 - i.
    if idx[0] >= 8 {
        std::mem::swap(&mut q0, &mut q1);
        std::mem::swap(&mut p0, &mut p1);
        for i in idx.iter_mut() {
            *i = 15 - *i;
        }
    }

    pack_mode6(&q0, &q1, p0, p1, &idx)
}

/// Quantize a float endpoint to 7 bits + shared p-bit, picking the p-bit that
/// minimizes total channel error. Decoded value per channel: `(q << 1) | p`.
fn quantize_endpoint(e: &[f32; 4]) -> ([u8; 4], u8) {
    let mut best = ([0u8; 4], 0u8);
    let mut best_err = f32::MAX;
    for p in [0u8, 1u8] {
        let mut q = [0u8; 4];
        let mut err = 0f32;
        for c in 0..4 {
            q[c] = ((e[c] - p as f32) / 2.0).round().clamp(0.0, 127.0) as u8;
            let recon = (q[c] * 2 + p) as f32;
            err += (recon - e[c]).powi(2);
        }
        if err < best_err {
            best_err = err;
            best = (q, p);
        }
    }
    best
}

fn reconstruct(q: &[u8; 4], p: u8) -> [f32; 4] {
    [
        (q[0] * 2 + p) as f32,
        (q[1] * 2 + p) as f32,
        (q[2] * 2 + p) as f32,
        (q[3] * 2 + p) as f32,
    ]
}

fn assign_indices(px: &[[u8; 4]; 16], q0: &[u8; 4], p0: u8, q1: &[u8; 4], p1: u8) -> [u8; 16] {
    let e0 = reconstruct(q0, p0);
    let e1 = reconstruct(q1, p1);
    let d = [e1[0] - e0[0], e1[1] - e0[1], e1[2] - e0[2], e1[3] - e0[3]];
    let len2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + d[3] * d[3];

    let mut idx = [0u8; 16];
    for (i, p) in px.iter().enumerate() {
        if len2 < 1e-6 {
            idx[i] = 0;
            continue;
        }
        let t: f32 = (0..4)
            .map(|c| (p[c] as f32 - e0[c]) * d[c])
            .sum::<f32>()
            / len2;
        // Nearest palette entry: compare t against w/64 for each weight.
        let mut best = 0u8;
        let mut best_err = f32::MAX;
        for (j, w) in WEIGHTS4.iter().enumerate() {
            let err = (*w as f32 / 64.0 - t).abs();
            if err < best_err {
                best_err = err;
                best = j as u8;
            }
        }
        idx[i] = best;
    }
    idx
}

/// Least-squares refit of both endpoints with the indices held fixed.
///
/// Minimizes `sum_i |(1 - w_i) * e0 + w_i * e1 - px_i|^2` per channel, which is
/// a 2x2 normal-equation solve shared by all four channels (they differ only in
/// the right-hand side). Returns `None` when the system is singular — every
/// pixel landed on the same index, so the two endpoints are not separable.
fn refit_endpoints(px: &[[u8; 4]; 16], idx: &[u8; 16]) -> Option<([f32; 4], [f32; 4])> {
    let (mut a, mut b, mut c) = (0f32, 0f32, 0f32);
    let mut rhs0 = [0f32; 4];
    let mut rhs1 = [0f32; 4];
    for (p, &i) in px.iter().zip(idx.iter()) {
        let w = WEIGHTS4[i as usize] as f32 / 64.0;
        let v = 1.0 - w;
        a += v * v;
        b += v * w;
        c += w * w;
        for ch in 0..4 {
            rhs0[ch] += v * p[ch] as f32;
            rhs1[ch] += w * p[ch] as f32;
        }
    }
    let det = a * c - b * b;
    if det.abs() < 1e-6 {
        return None;
    }
    let mut e0 = [0f32; 4];
    let mut e1 = [0f32; 4];
    for ch in 0..4 {
        e0[ch] = ((c * rhs0[ch] - b * rhs1[ch]) / det).clamp(0.0, 255.0);
        e1[ch] = ((a * rhs1[ch] - b * rhs0[ch]) / det).clamp(0.0, 255.0);
    }
    Some((e0, e1))
}

/// Total squared error of a block decoded from the given endpoints and indices.
///
/// Interpolates with the spec's integer arithmetic rather than in floats: the
/// decoder truncates, and a refit that looks better in exact arithmetic can
/// decode worse once rounding is applied. Refinement compares against this, so
/// it must agree with the real decoder.
fn block_error(
    px: &[[u8; 4]; 16],
    q0: &[u8; 4],
    p0: u8,
    q1: &[u8; 4],
    p1: u8,
    idx: &[u8; 16],
) -> u32 {
    let e0 = reconstruct(q0, p0);
    let e1 = reconstruct(q1, p1);
    let mut err = 0u32;
    for (p, &i) in px.iter().zip(idx.iter()) {
        let w = WEIGHTS4[i as usize] as i32;
        for ch in 0..4 {
            let recon = ((64 - w) * e0[ch] as i32 + w * e1[ch] as i32 + 32) >> 6;
            let d = recon - p[ch] as i32;
            err += (d * d) as u32;
        }
    }
    err
}

/// Pack a mode-6 block: mode bits, 8x7-bit endpoints, 2 p-bits, then the
/// anchor index in 3 bits and 15 indices in 4 bits, all LSB-first.
fn pack_mode6(q0: &[u8; 4], q1: &[u8; 4], p0: u8, p1: u8, idx: &[u8; 16]) -> [u8; 16] {
    let mut block: u128 = 1 << 6; // mode 6
    let mut pos = 7;
    for c in 0..4 {
        block |= (q0[c] as u128) << pos;
        block |= (q1[c] as u128) << (pos + 7);
        pos += 14;
    }
    block |= (p0 as u128) << 63;
    block |= (p1 as u128) << 64;
    let mut pos = 65;
    for (i, v) in idx.iter().enumerate() {
        let bits = if i == 0 { 3 } else { 4 };
        block |= ((*v as u128) & ((1 << bits) - 1)) << pos;
        pos += bits;
    }
    debug_assert_eq!(pos, 128);
    block.to_le_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference mode-6 decode, mirroring the BC7 spec.
    fn decode_block(block: &[u8; 16]) -> [[u8; 4]; 16] {
        let b = u128::from_le_bytes(*block);
        assert_eq!(b & 0x7F, 1 << 6, "expected mode 6");
        let mut pos = 7;
        let mut ep = [[0u8; 4]; 2];
        for c in 0..4 {
            ep[0][c] = ((b >> pos) & 0x7F) as u8;
            ep[1][c] = ((b >> (pos + 7)) & 0x7F) as u8;
            pos += 14;
        }
        let p0 = ((b >> 63) & 1) as u8;
        let p1 = ((b >> 64) & 1) as u8;
        let e0 = reconstruct(&ep[0], p0);
        let e1 = reconstruct(&ep[1], p1);

        let mut pos = 65;
        let mut out = [[0u8; 4]; 16];
        for i in 0..16 {
            let bits = if i == 0 { 3 } else { 4 };
            let idx = ((b >> pos) & ((1 << bits) - 1)) as usize;
            pos += bits;
            let w = WEIGHTS4[idx] as f32;
            for c in 0..4 {
                out[i][c] = (((64.0 - w) * e0[c] + w * e1[c] + 32.0) / 64.0) as u8;
            }
        }
        out
    }

    fn roundtrip_error(px: &[[u8; 4]; 16]) -> u8 {
        roundtrip_error_at(px, 1)
    }

    fn roundtrip_error_at(px: &[[u8; 4]; 16], refine_iters: u32) -> u8 {
        let block = encode_block(px, refine_iters);
        let dec = decode_block(&block);
        let mut max_err = 0u8;
        for i in 0..16 {
            for c in 0..4 {
                max_err = max_err.max(px[i][c].abs_diff(dec[i][c]));
            }
        }
        max_err
    }

    /// Total squared error of a full encode/decode round trip.
    fn sse_at(px: &[[u8; 4]; 16], refine_iters: u32) -> u64 {
        let dec = decode_block(&encode_block(px, refine_iters));
        let mut sse = 0u64;
        for i in 0..16 {
            for c in 0..4 {
                let d = px[i][c].abs_diff(dec[i][c]) as u64;
                sse += d * d;
            }
        }
        sse
    }

    /// Each refinement round is accepted only if it lowers the block error, so
    /// raising the iteration count must never raise the error.
    #[test]
    fn refinement_never_worsens_a_block() {
        // A few blocks with different structure: gradient, two-tone, noise-ish.
        let mut blocks: Vec<[[u8; 4]; 16]> = Vec::new();

        let mut grad = [[0u8; 4]; 16];
        for (i, p) in grad.iter_mut().enumerate() {
            *p = [(i * 16) as u8, 255 - (i * 16) as u8, 128, (i * 15) as u8];
        }
        blocks.push(grad);

        let mut two_tone = [[20u8, 30, 200, 255]; 16];
        for p in two_tone.iter_mut().take(7) {
            *p = [230, 40, 10, 90];
        }
        blocks.push(two_tone);

        let mut noisy = [[0u8; 4]; 16];
        for (i, p) in noisy.iter_mut().enumerate() {
            let v = ((i * 97 + 13) % 251) as u8;
            *p = [v, v.wrapping_mul(3), v.wrapping_add(77), 255];
        }
        blocks.push(noisy);

        for px in &blocks {
            let e0 = sse_at(px, 0);
            let e1 = sse_at(px, 1);
            let e4 = sse_at(px, 4);
            assert!(e1 <= e0, "1 iter worse than 0: {e1} > {e0}");
            assert!(e4 <= e1, "4 iters worse than 1: {e4} > {e1}");
        }
    }

    /// Refinement has to actually buy something, or the quality dial is
    /// decorative. Mode 6 fits one line through the block, so the gain is
    /// modest on average (about 0.4% of total squared error over random
    /// blocks); this is a block where the PCA extremes overshoot badly and
    /// least squares pulls them back.
    #[test]
    fn refinement_improves_a_scattered_block() {
        let px = [
            [151u8, 34, 95, 255], [49, 63, 190, 255], [61, 32, 82, 255], [247, 153, 234, 255],
            [189, 141, 93, 255], [200, 198, 173, 255], [50, 8, 236, 255], [167, 247, 169, 255],
            [41, 16, 251, 255], [150, 56, 48, 255], [103, 38, 124, 255], [223, 207, 169, 255],
            [182, 184, 28, 255], [211, 169, 65, 255], [46, 9, 9, 255], [56, 196, 184, 255],
        ];
        let plain = sse_at(&px, 0);
        let refined = sse_at(&px, 6);
        assert!(
            refined < plain * 9 / 10,
            "expected a clear improvement, got {plain} -> {refined}"
        );
    }

    #[test]
    fn solid_color_is_near_exact() {
        let px = [[37u8, 200, 90, 255]; 16];
        assert!(roundtrip_error(&px) <= 1);
    }

    #[test]
    fn grayscale_gradient() {
        let mut px = [[0u8; 4]; 16];
        for (i, p) in px.iter_mut().enumerate() {
            let v = (i * 17) as u8;
            *p = [v, v, v, 255];
        }
        assert!(roundtrip_error(&px) <= 8);
    }

    #[test]
    fn color_gradient_with_alpha() {
        let mut px = [[0u8; 4]; 16];
        for (i, p) in px.iter_mut().enumerate() {
            *p = [(i * 16) as u8, 255 - (i * 16) as u8, 128, (i * 15) as u8];
        }
        assert!(roundtrip_error(&px) <= 12);
    }

    #[test]
    fn anchor_swap_preserves_decode() {
        // Bright pixel at position 0 forces the anchor index into the top half
        // unless endpoints are swapped.
        let mut px = [[10u8, 10, 10, 255]; 16];
        px[0] = [250, 240, 230, 255];
        let block = encode_block(&px, 1);
        // Anchor index (3 bits at position 65) may be anything valid; decode
        // must still approximate pixel 0 well.
        let dec = decode_block(&block);
        assert!(dec[0][0].abs_diff(250) <= 12);
    }
}
