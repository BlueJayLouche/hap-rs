// BC7 (mode 6) GPU Compression Shader
//
// Compresses RGBA pixels to BC7 format (16 bytes per 4x4 block).
// Each invocation processes one 4x4 block.
//
// Algorithm (same as hap_qt::bc7): PCA endpoints (power iteration over the
// 4D covariance), 7-bit + p-bit endpoint quantization, 4-bit indices by
// nearest palette entry, `refine_iters` least-squares endpoint refits, and an
// endpoint swap when the anchor index >= 8.
//
// Bit layout (LSB-first over 128 bits):
//   bits 0-6:   mode = 1<<6
//   bits 7-62:  R0,R1,G0,G1,B0,B1,A0,A1 (7 bits each)
//   bits 63-64: P0, P1
//   bits 65+:   anchor index in 3 bits, then 15 indices in 4 bits


const WEIGHTS4: array<f32, 16> = array<f32, 16>(
    0.0, 4.0, 9.0, 13.0, 17.0, 21.0, 26.0, 30.0,
    34.0, 38.0, 43.0, 47.0, 51.0, 55.0, 60.0, 64.0
);

// Write `nbits` of `value` at absolute bit position `pos`, LSB-first.
fn put_bits(out: ptr<function, array<u32, 4>>, pos: u32, value_in: u32, nbits: u32) {
    let value = value_in & ((1u << nbits) - 1u);
    let w = pos >> 5u;
    let off = pos & 31u;
    (*out)[w] = (*out)[w] | (value << off);
    if off + nbits > 32u {
        (*out)[w + 1u] = (*out)[w + 1u] | (value >> (32u - off));
    }
}

// Quantize a float endpoint to 7 bits + shared p-bit (best p wins).
// Returns (q << 1) | p packed per channel in the low 8 bits; p via out param.
fn quantize_endpoint(e: vec4<f32>, p_out: ptr<function, u32>) -> vec4<u32> {
    var best_q = vec4<u32>(0u);
    var best_p = 0u;
    var best_err = 1e30;
    for (var p = 0u; p <= 1u; p = p + 1u) {
        let pf = vec4<f32>(f32(p));
        let q = vec4<u32>(clamp(round((e - pf) / 2.0), vec4<f32>(0.0), vec4<f32>(127.0)));
        let recon = vec4<f32>(q * 2u + vec4<u32>(p));
        let d = recon - e;
        let err = dot(d, d);
        if err < best_err {
            best_err = err;
            best_q = q;
            best_p = p;
        }
    }
    *p_out = best_p;
    return best_q;
}

var<private> bc7_px: array<vec4<f32>, 16>;
var<private> bc7_idx: array<u32, 16>;
var<private> bc7_best_idx: array<u32, 16>;

// Assign every pixel its nearest palette entry for the given quantized
// endpoints, leaving the result in bc7_idx, and return the block error.
//
// The error uses the decoder's integer interpolation rather than exact
// arithmetic: the decoder truncates, and a refit that looks better in floats
// can decode worse. Refinement accepts rounds based on this, so it has to
// agree with the real decoder.
fn bc7_assign(q0: vec4<u32>, p0: u32, q1: vec4<u32>, p1: u32) -> f32 {
    let r0 = vec4<f32>(q0 * 2u + vec4<u32>(p0));
    let r1 = vec4<f32>(q1 * 2u + vec4<u32>(p1));
    let d = r1 - r0;
    let len2 = dot(d, d);

    var err = 0.0;
    for (var i = 0u; i < 16u; i = i + 1u) {
        var best = 0u;
        if len2 >= 1e-6 {
            let t = dot(bc7_px[i] - r0, d) / len2;
            var best_err = 1e30;
            for (var j = 0u; j < 16u; j = j + 1u) {
                let e = abs(WEIGHTS4[j] / 64.0 - t);
                if e < best_err {
                    best_err = e;
                    best = j;
                }
            }
        }
        bc7_idx[i] = best;

        let w = WEIGHTS4[best];
        let recon = floor(((64.0 - w) * r0 + w * r1 + 32.0) / 64.0);
        let diff = recon - bc7_px[i];
        err = err + dot(diff, diff);
    }
    return err;
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let block_x = gid.x;
    let block_y = gid.y;
    if block_x >= params.blocks_x || block_y >= params.blocks_y {
        return;
    }

    // Load 16 pixels
    let px_x = block_x * 4u;
    let px_y = block_y * 4u;
    for (var y = 0u; y < 4u; y = y + 1u) {
        for (var x = 0u; x < 4u; x = x + 1u) {
            let idx = (px_y + y) * params.width + (px_x + x);
            bc7_px[y * 4u + x] = unpack_rgba(input_pixels[idx]);
        }
    }

    // Mean and 4x4 covariance
    var mean = vec4<f32>(0.0);
    for (var i = 0u; i < 16u; i = i + 1u) {
        mean = mean + bc7_px[i];
    }
    mean = mean / 16.0;

    var cov: array<vec4<f32>, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        cov[i] = vec4<f32>(0.0);
    }
    for (var i = 0u; i < 16u; i = i + 1u) {
        let d = bc7_px[i] - mean;
        cov[0] = cov[0] + d.x * d;
        cov[1] = cov[1] + d.y * d;
        cov[2] = cov[2] + d.z * d;
        cov[3] = cov[3] + d.w * d;
    }

    // Seed axis = max-range channel
    var mn = vec4<f32>(255.0);
    var mx = vec4<f32>(0.0);
    for (var i = 0u; i < 16u; i = i + 1u) {
        mn = min(mn, bc7_px[i]);
        mx = max(mx, bc7_px[i]);
    }
    let range = mx - mn;
    var axis = vec4<f32>(0.0);
    if range.x >= range.y && range.x >= range.z && range.x >= range.w {
        axis.x = 1.0;
    } else if range.y >= range.z && range.y >= range.w {
        axis.y = 1.0;
    } else if range.z >= range.w {
        axis.z = 1.0;
    } else {
        axis.w = 1.0;
    }

    // Power iteration for the principal axis
    for (var it = 0u; it < 10u; it = it + 1u) {
        var v = vec4<f32>(0.0);
        v.x = dot(cov[0], axis);
        v.y = dot(cov[1], axis);
        v.z = dot(cov[2], axis);
        v.w = dot(cov[3], axis);
        let n = sqrt(dot(v, v));
        if n < 1e-6 {
            break;
        }
        axis = v / n;
    }

    // Endpoints at the projection extremes
    var t_min = 1e30;
    var t_max = -1e30;
    for (var i = 0u; i < 16u; i = i + 1u) {
        let t = dot(bc7_px[i] - mean, axis);
        t_min = min(t_min, t);
        t_max = max(t_max, t);
    }
    let e0 = clamp(mean + t_min * axis, vec4<f32>(0.0), vec4<f32>(255.0));
    let e1 = clamp(mean + t_max * axis, vec4<f32>(0.0), vec4<f32>(255.0));

    // Quantize endpoints to 7 bits + p-bit, then index against them.
    var p0 = 0u;
    var p1 = 0u;
    var q0 = quantize_endpoint(e0, &p0);
    var q1 = quantize_endpoint(e1, &p1);

    var best_err = bc7_assign(q0, p0, q1, p1);
    for (var i = 0u; i < 16u; i = i + 1u) {
        bc7_best_idx[i] = bc7_idx[i];
    }

    // Least-squares refinement: refit the endpoints to the indices just
    // assigned, then re-index. A round is kept only if it lowers the block
    // error, so more iterations can never make a block worse.
    for (var it = 0u; it < params.refine_iters; it = it + 1u) {
        for (var i = 0u; i < 16u; i = i + 1u) {
            fit_px[i] = bc7_px[i];
            fit_w[i] = WEIGHTS4[bc7_best_idx[i]] / 64.0;
        }
        let cur = Endpoints(
            vec4<f32>(q0 * 2u + vec4<u32>(p0)),
            vec4<f32>(q1 * 2u + vec4<u32>(p1)),
        );
        let r = refit_endpoints(cur);

        var np0 = 0u;
        var np1 = 0u;
        let nq0 = quantize_endpoint(r.e0, &np0);
        let nq1 = quantize_endpoint(r.e1, &np1);
        let err = bc7_assign(nq0, np0, nq1, np1);
        if err >= best_err {
            break; // converged
        }
        best_err = err;
        q0 = nq0;
        p0 = np0;
        q1 = nq1;
        p1 = np1;
        for (var i = 0u; i < 16u; i = i + 1u) {
            bc7_best_idx[i] = bc7_idx[i];
        }
    }

    var idx: array<u32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        idx[i] = bc7_best_idx[i];
    }

    // Anchor index (pixel 0) is stored in 3 bits; swap endpoints if needed.
    // Weights satisfy w[15-i] = 64 - w[i], so the swap is exact.
    if idx[0] >= 8u {
        let tq = q0;
        q0 = q1;
        q1 = tq;
        let tp = p0;
        p0 = p1;
        p1 = tp;
        for (var i = 0u; i < 16u; i = i + 1u) {
            idx[i] = 15u - idx[i];
        }
    }

    // Pack 128 bits LSB-first into 4 u32 words
    var words = array<u32, 4>(0u, 0u, 0u, 0u);
    put_bits(&words, 0u, 64u, 7u); // mode 6
    var pos = 7u;
    for (var c = 0u; c < 4u; c = c + 1u) {
        put_bits(&words, pos, q0[c], 7u);
        put_bits(&words, pos + 7u, q1[c], 7u);
        pos = pos + 14u;
    }
    put_bits(&words, 63u, p0, 1u);
    put_bits(&words, 64u, p1, 1u);
    pos = 65u;
    for (var i = 0u; i < 16u; i = i + 1u) {
        var bits = 4u;
        if i == 0u {
            bits = 3u;
        }
        put_bits(&words, pos, idx[i], bits);
        pos = pos + bits;
    }

    let out_idx = (block_y * params.blocks_x + block_x) * 4u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        output_blocks[out_idx + i] = words[i];
    }
}
