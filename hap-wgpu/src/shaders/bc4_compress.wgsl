// BC4/RGTC1 GPU Compression Shader (HapA / Alpha-only format)
//
// Compresses single-channel data to BC4 format (8 bytes per 4x4 block).
// Extracts alpha channel from RGBA input.

struct Params {
    width: u32,
    height: u32,
    blocks_x: u32,
    blocks_y: u32,
}

@group(0) @binding(0) var<storage, read> input_pixels: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_blocks: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let block_x = gid.x;
    let block_y = gid.y;

    if block_x >= params.blocks_x || block_y >= params.blocks_y {
        return;
    }

    // Load 16 alpha values
    var alphas: array<f32, 16>;
    var alpha_min = 255.0;
    var alpha_max = 0.0;

    let px_x = block_x * 4u;
    let px_y = block_y * 4u;

    for (var y = 0u; y < 4u; y = y + 1u) {
        for (var x = 0u; x < 4u; x = x + 1u) {
            let sx = px_x + x;
            let sy = px_y + y;
            let idx = sy * params.width + sx;
            let packed = input_pixels[idx];
            let alpha = f32((packed >> 24u) & 0xFFu);
            let pi = y * 4u + x;
            alphas[pi] = alpha;
            alpha_min = min(alpha_min, alpha);
            alpha_max = max(alpha_max, alpha);
        }
    }

    let a0 = u32(clamp(alpha_max, 0.0, 255.0));
    let a1 = u32(clamp(alpha_min, 0.0, 255.0));

    // Build 8-level palette (a0 > a1 mode)
    var palette: array<f32, 8>;
    palette[0] = f32(a0);
    palette[1] = f32(a1);
    if a0 > a1 {
        palette[2] = (6.0 * f32(a0) + 1.0 * f32(a1)) / 7.0;
        palette[3] = (5.0 * f32(a0) + 2.0 * f32(a1)) / 7.0;
        palette[4] = (4.0 * f32(a0) + 3.0 * f32(a1)) / 7.0;
        palette[5] = (3.0 * f32(a0) + 4.0 * f32(a1)) / 7.0;
        palette[6] = (2.0 * f32(a0) + 5.0 * f32(a1)) / 7.0;
        palette[7] = (1.0 * f32(a0) + 6.0 * f32(a1)) / 7.0;
    } else {
        palette[2] = (4.0 * f32(a0) + 1.0 * f32(a1)) / 5.0;
        palette[3] = (3.0 * f32(a0) + 2.0 * f32(a1)) / 5.0;
        palette[4] = (2.0 * f32(a0) + 3.0 * f32(a1)) / 5.0;
        palette[5] = (1.0 * f32(a0) + 4.0 * f32(a1)) / 5.0;
        palette[6] = 0.0;
        palette[7] = 255.0;
    }

    // Encode 16 pixels as 3-bit indices
    var indices_lo = 0u;
    var indices_hi = 0u;

    for (var i = 0u; i < 16u; i = i + 1u) {
        var best_idx = 0u;
        var best_dist = abs(alphas[i] - palette[0]);
        for (var j = 1u; j < 8u; j = j + 1u) {
            let d = abs(alphas[i] - palette[j]);
            if d < best_dist {
                best_idx = j;
                best_dist = d;
            }
        }

        let bit_pos = i * 3u;
        if bit_pos < 32u {
            indices_lo = indices_lo | (best_idx << bit_pos);
            if bit_pos > 29u {
                indices_hi = indices_hi | (best_idx >> (32u - bit_pos));
            }
        } else {
            indices_hi = indices_hi | (best_idx << (bit_pos - 32u));
        }
    }

    // Write output: 2 u32 values = 8 bytes per block
    let block_idx = (block_y * params.blocks_x + block_x) * 2u;
    let word0 = a0 | (a1 << 8u) | ((indices_lo & 0xFFFFu) << 16u);
    let word1 = (indices_lo >> 16u) | (indices_hi << 16u);
    output_blocks[block_idx] = word0;
    output_blocks[block_idx + 1u] = word1;
}
