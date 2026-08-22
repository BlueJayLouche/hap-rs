// Shared endpoint-refinement helpers, prepended to every compression shader.
//
// All the block formats here encode a 4x4 block as two endpoints plus a
// per-pixel index into a palette interpolated between them. Once indices are
// assigned, the best possible endpoints are the least-squares solution, which
// is the same 2x2 normal-equation solve for every format — only the number of
// channels and the index->weight table differ. Iterating (refit, re-index)
// is what separates the quality presets from the plain bounding-box fit.
//
// Scratch lives in private vars rather than function parameters: WGSL passes
// arrays by value, and copying 16 vec4s per call per iteration is not free.

// Shared across every compression shader. All-u32 members keep the uniform
// layout flat; the padding is what makes the struct a multiple of 16 bytes,
// which uniform buffers require. Mirrors CompressParams in gpu_compress.rs.
struct Params {
    width: u32,
    height: u32,
    blocks_x: u32,
    blocks_y: u32,
    refine_iters: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input_pixels: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_blocks: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Endpoints {
    e0: vec4<f32>,
    e1: vec4<f32>,
}

// Block pixels, and the interpolation weight (0 = e0, 1 = e1) each one landed
// on. Fill both before calling refit_endpoints. Unused channels stay 0.
var<private> fit_px: array<vec4<f32>, 16>;
var<private> fit_w: array<f32, 16>;

// Least-squares refit of both endpoints with the indices held fixed:
// minimizes sum_i |(1 - w_i) * e0 + w_i * e1 - px_i|^2. All channels share one
// solve; they differ only in the right-hand side. Returns `cur` unchanged when
// the system is singular, which means every pixel landed on the same index and
// the two endpoints are not separable.
fn refit_endpoints(cur: Endpoints) -> Endpoints {
    var a = 0.0;
    var b = 0.0;
    var c = 0.0;
    var r0 = vec4<f32>(0.0);
    var r1 = vec4<f32>(0.0);

    for (var i = 0u; i < 16u; i = i + 1u) {
        let w = fit_w[i];
        let v = 1.0 - w;
        a = a + v * v;
        b = b + v * w;
        c = c + w * w;
        r0 = r0 + v * fit_px[i];
        r1 = r1 + w * fit_px[i];
    }

    let det = a * c - b * b;
    if abs(det) < 1e-6 {
        return cur;
    }
    let lo = vec4<f32>(0.0);
    let hi = vec4<f32>(255.0);
    return Endpoints(
        clamp((c * r0 - b * r1) / det, lo, hi),
        clamp((a * r1 - b * r0) / det, lo, hi),
    );
}


// ---------------------------------------------------------------------------
// RGB565 colour block (BC1, and the colour half of BC3 / YCoCg-BC3)
// ---------------------------------------------------------------------------

fn unpack_rgba(packed: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(packed & 0xFFu),
        f32((packed >> 8u) & 0xFFu),
        f32((packed >> 16u) & 0xFFu),
        f32((packed >> 24u) & 0xFFu)
    );
}

fn rgb_to_565(r: f32, g: f32, b: f32) -> u32 {
    let r5 = u32(clamp(r, 0.0, 255.0)) >> 3u;
    let g6 = u32(clamp(g, 0.0, 255.0)) >> 2u;
    let b5 = u32(clamp(b, 0.0, 255.0)) >> 3u;
    return (r5 << 11u) | (g6 << 5u) | b5;
}

fn rgb565_to_rgb(c: u32) -> vec3<f32> {
    let r5 = (c >> 11u) & 0x1Fu;
    let g6 = (c >> 5u) & 0x3Fu;
    let b5 = c & 0x1Fu;
    return vec3<f32>(
        f32((r5 << 3u) | (r5 >> 2u)),
        f32((g6 << 2u) | (g6 >> 4u)),
        f32((b5 << 3u) | (b5 >> 2u))
    );
}

fn color_dist_sq(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let d = a - b;
    return dot(d, d);
}

// Move one RGB565 channel by `delta` quantization steps.
fn nudge_565(c: u32, channel: u32, delta: i32) -> u32 {
    var r5 = i32((c >> 11u) & 0x1Fu);
    var g6 = i32((c >> 5u) & 0x3Fu);
    var b5 = i32(c & 0x1Fu);
    if channel == 0u {
        r5 = clamp(r5 + delta, 0, 31);
    } else if channel == 1u {
        g6 = clamp(g6 + delta, 0, 63);
    } else {
        b5 = clamp(b5 + delta, 0, 31);
    }
    return (u32(r5) << 11u) | (u32(g6) << 5u) | u32(b5);
}

// Interpolation weight of each index: 0 -> color0, 1 -> color1, 2 and 3 -> one
// and two thirds of the way toward color1.
fn color_weight(idx: u32) -> f32 {
    if idx == 0u { return 0.0; }
    if idx == 1u { return 1.0; }
    if idx == 2u { return 1.0 / 3.0; }
    return 2.0 / 3.0;
}

struct ColorFit {
    color0: u32,
    color1: u32,
    indices: u32,
    err: f32,
}

var<private> fit_pixels: array<vec3<f32>, 16>;

// When >= 0, both endpoints' blue channel is pinned to this RGB565 b5 code.
// Hap Q needs it: in a scaled-YCoCg colour block blue is not colour, it is the
// per-block chroma scale indicator, and the decoder recovers
// `scale = blue / 8 + 1`. Letting refit or polish drift it would rescale every
// pixel in the block.
var<private> fit_lock_b5: i32 = -1;

fn apply_b5_lock(c: u32) -> u32 {
    if fit_lock_b5 < 0 {
        return c;
    }
    return (c & 0xFFE0u) | u32(fit_lock_b5);
}

// Assign every pixel its nearest palette entry for a given endpoint pair, and
// report the resulting squared error. Reads fit_pixels.
fn color_fit_565(c0_in: u32, c1_in: u32) -> ColorFit {
    var color0 = apply_b5_lock(c0_in);
    var color1 = apply_b5_lock(c1_in);

    // color0 > color1 selects 4-colour mode (no punch-through alpha).
    if color0 < color1 {
        let tmp = color0;
        color0 = color1;
        color1 = tmp;
    }
    if color0 == color1 {
        // Endpoints must differ to stay in 4-colour mode. Perturb green rather
        // than the low bits, which would disturb a locked blue.
        let g6 = i32((color0 >> 5u) & 0x3Fu);
        color1 = nudge_565(color0, 1u, select(1, -1, g6 > 0));
        if color0 < color1 {
            let tmp = color0;
            color0 = color1;
            color1 = tmp;
        }
    }

    let p0 = rgb565_to_rgb(color0);
    let p1 = rgb565_to_rgb(color1);
    let p2 = (2.0 * p0 + p1) / 3.0;
    let p3 = (p0 + 2.0 * p1) / 3.0;

    var indices = 0u;
    var err = 0.0;
    for (var i = 0u; i < 16u; i = i + 1u) {
        let px = fit_pixels[i];
        let d0 = color_dist_sq(px, p0);
        let d1 = color_dist_sq(px, p1);
        let d2 = color_dist_sq(px, p2);
        let d3 = color_dist_sq(px, p3);

        var best_idx = 0u;
        var best_dist = d0;
        if d1 < best_dist { best_idx = 1u; best_dist = d1; }
        if d2 < best_dist { best_idx = 2u; best_dist = d2; }
        if d3 < best_dist { best_idx = 3u; best_dist = d3; }

        indices = indices | (best_idx << (i * 2u));
        err = err + best_dist;
    }
    return ColorFit(color0, color1, indices, err);
}

// Least-squares refit rounds. Fixes endpoint *placement* on gradient and
// detail blocks, where the bounding box overshoots.
fn color_refine(initial: ColorFit) -> ColorFit {
    var best = initial;
    for (var it = 0u; it < params.refine_iters; it = it + 1u) {
        for (var i = 0u; i < 16u; i = i + 1u) {
            fit_px[i] = vec4<f32>(fit_pixels[i], 0.0);
            fit_w[i] = color_weight((best.indices >> (i * 2u)) & 3u);
        }
        let cur = Endpoints(
            vec4<f32>(rgb565_to_rgb(best.color0), 0.0),
            vec4<f32>(rgb565_to_rgb(best.color1), 0.0),
        );
        let r = refit_endpoints(cur);
        let cand = color_fit_565(rgb_to_565(r.e0.x, r.e0.y, r.e0.z),
                                 rgb_to_565(r.e1.x, r.e1.y, r.e1.z));
        if cand.err >= best.err {
            break; // converged
        }
        best = cand;
    }
    return best;
}

// Hill-climb the quantized endpoints one RGB565 step at a time.
//
// This is what recovers flat and two-colour blocks. There the least-squares
// refit is singular (every pixel lands on one index) and the limit is RGB565
// quantization, not endpoint placement: the fix is to pick an endpoint pair
// whose 1/3 or 2/3 interpolant lands nearer the target than either endpoint
// can. Only strict improvements are kept, so this terminates.
fn color_polish(initial: ColorFit) -> ColorFit {
    var best = initial;
    for (var round = 0u; round < params.refine_iters; round = round + 1u) {
        var improved = false;
        // A locked blue cannot be improved, so do not spend candidates on it.
        let channels = select(3u, 2u, fit_lock_b5 >= 0);
        for (var e = 0u; e < 2u; e = e + 1u) {
            for (var ch = 0u; ch < channels; ch = ch + 1u) {
                for (var d = 0u; d < 2u; d = d + 1u) {
                    let delta = select(-1, 1, d == 0u);
                    var c0 = best.color0;
                    var c1 = best.color1;
                    if e == 0u {
                        c0 = nudge_565(c0, ch, delta);
                    } else {
                        c1 = nudge_565(c1, ch, delta);
                    }
                    let cand = color_fit_565(c0, c1);
                    if cand.err < best.err {
                        best = cand;
                        improved = true;
                    }
                }
            }
        }
        if !improved {
            break;
        }
    }
    return best;
}

// Full colour-block encode: bounding box, then refit, then polish.
// Caller fills fit_pixels first.
fn encode_color_block() -> ColorFit {
    var lo = vec3<f32>(255.0);
    var hi = vec3<f32>(0.0);
    for (var i = 0u; i < 16u; i = i + 1u) {
        lo = min(lo, fit_pixels[i]);
        hi = max(hi, fit_pixels[i]);
    }
    // Inset by 1/16: trades a little accuracy on flat blocks for a better fit
    // on gradients. Polishing wins that back.
    let inset = (hi - lo) / 16.0;
    let a = clamp(hi - inset, vec3<f32>(0.0), vec3<f32>(255.0));
    let b = clamp(lo + inset, vec3<f32>(0.0), vec3<f32>(255.0));

    let base = color_fit_565(rgb_to_565(a.x, a.y, a.z), rgb_to_565(b.x, b.y, b.z));
    return color_polish(color_refine(base));
}


// ---------------------------------------------------------------------------
// 8-level scalar block (BC4, and the alpha half of BC3 / the Y plane of Hap Q)
// ---------------------------------------------------------------------------

var<private> fit_scalars: array<f32, 16>;

struct AlphaFit {
    a0: u32,
    a1: u32,
    words: vec2<u32>,
    err: f32,
}

// Interpolation weight of each index in a0 > a1 mode: 0 -> a0, 1 -> a1,
// 2..7 -> one through six sevenths of the way toward a1.
fn alpha_weight(idx: u32) -> f32 {
    if idx == 0u { return 0.0; }
    if idx == 1u { return 1.0; }
    return f32(idx - 1u) / 7.0;
}

// Assign every value its nearest palette entry and pack the 3-bit indices.
// Always uses the 8-level (a0 > a1) mode. Reads fit_scalars.
fn alpha_fit(a0_in: u32, a1_in: u32) -> AlphaFit {
    var a0 = a0_in;
    var a1 = a1_in;
    if a0 < a1 {
        let t = a0;
        a0 = a1;
        a1 = t;
    }

    var palette: array<f32, 8>;
    palette[0] = f32(a0);
    palette[1] = f32(a1);
    for (var j = 2u; j < 8u; j = j + 1u) {
        let w = f32(j - 1u) / 7.0;
        palette[j] = mix(f32(a0), f32(a1), w);
    }

    var indices_lo = 0u;
    var indices_hi = 0u;
    var err = 0.0;

    for (var i = 0u; i < 16u; i = i + 1u) {
        var best_idx = 0u;
        var best_dist = abs(fit_scalars[i] - palette[0]);
        for (var j = 1u; j < 8u; j = j + 1u) {
            let d = abs(fit_scalars[i] - palette[j]);
            if d < best_dist {
                best_idx = j;
                best_dist = d;
            }
        }
        err = err + best_dist * best_dist;

        let bit_pos = i * 3u;
        if bit_pos < 32u {
            indices_lo = indices_lo | (best_idx << bit_pos);
            // Handle bits that cross the 32-bit boundary
            if bit_pos > 29u {
                indices_hi = indices_hi | (best_idx >> (32u - bit_pos));
            }
        } else {
            indices_hi = indices_hi | (best_idx << (bit_pos - 32u));
        }
    }

    // Pack: first u32 = [alpha0:8][alpha1:8][indices bits 0-15]
    //        second u32 = [indices bits 16-47]
    let word0 = a0 | (a1 << 8u) | ((indices_lo & 0xFFFFu) << 16u);
    let word1 = (indices_lo >> 16u) | (indices_hi << 16u);
    return AlphaFit(a0, a1, vec2<u32>(word0, word1), err);
}

// Full scalar-block encode: min/max endpoints, then least-squares refit, then
// a one-step hill climb, mirroring the colour path.
fn encode_alpha_block_from_scalars() -> AlphaFit {
    var lo = fit_scalars[0];
    var hi = fit_scalars[0];
    for (var i = 1u; i < 16u; i = i + 1u) {
        lo = min(lo, fit_scalars[i]);
        hi = max(hi, fit_scalars[i]);
    }
    var best = alpha_fit(u32(clamp(hi, 0.0, 255.0)), u32(clamp(lo, 0.0, 255.0)));

    for (var it = 0u; it < params.refine_iters; it = it + 1u) {
        // Unpack the 48-bit index stream: it starts 16 bits into word0 and
        // continues through word1.
        let ilo = (best.words.x >> 16u) | (best.words.y << 16u); // index bits 0..31
        let ihi = best.words.y >> 16u;                           // index bits 32..47
        for (var i = 0u; i < 16u; i = i + 1u) {
            let bit_pos = i * 3u;
            var idx = 0u;
            if bit_pos >= 32u {
                idx = ihi >> (bit_pos - 32u);
            } else {
                idx = ilo >> bit_pos;
                if bit_pos > 29u {
                    idx = idx | (ihi << (32u - bit_pos));
                }
            }
            fit_px[i] = vec4<f32>(fit_scalars[i], 0.0, 0.0, 0.0);
            fit_w[i] = alpha_weight(idx & 7u);
        }
        let cur = Endpoints(vec4<f32>(f32(best.a0), 0.0, 0.0, 0.0),
                            vec4<f32>(f32(best.a1), 0.0, 0.0, 0.0));
        let r = refit_endpoints(cur);
        let cand = alpha_fit(u32(round(clamp(r.e0.x, 0.0, 255.0))),
                             u32(round(clamp(r.e1.x, 0.0, 255.0))));
        if cand.err >= best.err {
            break;
        }
        best = cand;
    }

    // Hill climb both endpoints by one code at a time.
    for (var round_i = 0u; round_i < params.refine_iters; round_i = round_i + 1u) {
        var improved = false;
        for (var e = 0u; e < 2u; e = e + 1u) {
            for (var d = 0u; d < 2u; d = d + 1u) {
                let delta = select(-1, 1, d == 0u);
                var a0 = i32(best.a0);
                var a1 = i32(best.a1);
                if e == 0u { a0 = clamp(a0 + delta, 0, 255); } else { a1 = clamp(a1 + delta, 0, 255); }
                let cand = alpha_fit(u32(a0), u32(a1));
                if cand.err < best.err {
                    best = cand;
                    improved = true;
                }
            }
        }
        if !improved {
            break;
        }
    }
    return best;
}
