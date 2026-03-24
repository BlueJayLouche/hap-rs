// BC3/DXT5 GPU Compression Shader
//
// Compresses RGBA pixels to BC3 format (16 bytes per 4x4 block).
// BC3 = alpha block (8 bytes) + BC1 color block (8 bytes).
// Each invocation processes one 4x4 block.

struct Params {
    width: u32,
    height: u32,
    blocks_x: u32,
    blocks_y: u32,
}

@group(0) @binding(0) var<storage, read> input_pixels: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_blocks: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

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

// Encode the BC3 alpha block: 2 endpoint bytes + 6 bytes of 3-bit indices
// Returns two u32 values (8 bytes)
fn encode_alpha_block(alphas: array<f32, 16>) -> vec2<u32> {
    // Find min/max alpha
    var alpha_min = alphas[0];
    var alpha_max = alphas[0];
    for (var i = 1u; i < 16u; i = i + 1u) {
        alpha_min = min(alpha_min, alphas[i]);
        alpha_max = max(alpha_max, alphas[i]);
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

    // Encode 16 pixels as 3-bit indices (48 bits total)
    var indices_lo = 0u; // bits 0..31
    var indices_hi = 0u; // bits 32..47 (only lower 16 bits used)

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

    return vec2<u32>(word0, word1);
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let block_x = gid.x;
    let block_y = gid.y;

    if block_x >= params.blocks_x || block_y >= params.blocks_y {
        return;
    }

    // Load 16 pixels
    var rgb_pixels: array<vec3<f32>, 16>;
    var alpha_values: array<f32, 16>;
    var min_color = vec3<f32>(255.0, 255.0, 255.0);
    var max_color = vec3<f32>(0.0, 0.0, 0.0);

    let px_x = block_x * 4u;
    let px_y = block_y * 4u;

    for (var y = 0u; y < 4u; y = y + 1u) {
        for (var x = 0u; x < 4u; x = x + 1u) {
            let sx = px_x + x;
            let sy = px_y + y;
            let idx = sy * params.width + sx;
            let rgba = unpack_rgba(input_pixels[idx]);
            let pi = y * 4u + x;
            rgb_pixels[pi] = rgba.xyz;
            alpha_values[pi] = rgba.w;
            min_color = min(min_color, rgba.xyz);
            max_color = max(max_color, rgba.xyz);
        }
    }

    // === Alpha block (8 bytes) ===
    let alpha_result = encode_alpha_block(alpha_values);

    // === Color block (8 bytes) - same as BC1 ===
    let inset = (max_color - min_color) / 16.0;
    let adj_min = clamp(min_color + inset, vec3<f32>(0.0), vec3<f32>(255.0));
    let adj_max = clamp(max_color - inset, vec3<f32>(0.0), vec3<f32>(255.0));

    var color0 = rgb_to_565(adj_max.x, adj_max.y, adj_max.z);
    var color1 = rgb_to_565(adj_min.x, adj_min.y, adj_min.z);

    if color0 < color1 {
        let tmp = color0;
        color0 = color1;
        color1 = tmp;
    }
    if color0 == color1 {
        if color0 > 0u { color1 = color0 - 1u; } else { color0 = 1u; }
    }

    let p0 = rgb565_to_rgb(color0);
    let p1 = rgb565_to_rgb(color1);
    let p2 = (2.0 * p0 + p1) / 3.0;
    let p3 = (p0 + 2.0 * p1) / 3.0;

    var color_indices = 0u;
    for (var i = 0u; i < 16u; i = i + 1u) {
        let px = rgb_pixels[i];
        let d0 = color_dist_sq(px, p0);
        let d1 = color_dist_sq(px, p1);
        let d2 = color_dist_sq(px, p2);
        let d3 = color_dist_sq(px, p3);

        var best_idx = 0u;
        var best_dist = d0;
        if d1 < best_dist { best_idx = 1u; best_dist = d1; }
        if d2 < best_dist { best_idx = 2u; best_dist = d2; }
        if d3 < best_dist { best_idx = 3u; }

        color_indices = color_indices | (best_idx << (i * 2u));
    }

    // Write output: 4 u32 values = 16 bytes per block
    // Layout: [alpha_word0][alpha_word1][color0|color1][color_indices]
    let block_idx = (block_y * params.blocks_x + block_x) * 4u;
    output_blocks[block_idx] = alpha_result.x;
    output_blocks[block_idx + 1u] = alpha_result.y;
    output_blocks[block_idx + 2u] = color0 | (color1 << 16u);
    output_blocks[block_idx + 3u] = color_indices;
}
