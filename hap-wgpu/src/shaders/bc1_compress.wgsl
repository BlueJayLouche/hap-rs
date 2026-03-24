// BC1/DXT1 GPU Compression Shader
//
// Compresses RGBA pixels to BC1 format (8 bytes per 4x4 block).
// Each invocation processes one 4x4 block.
//
// Algorithm: Bounding-box endpoint selection with RGB565 encoding.

struct Params {
    width: u32,
    height: u32,
    blocks_x: u32,
    blocks_y: u32,
}

@group(0) @binding(0) var<storage, read> input_pixels: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_blocks: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

// Unpack RGBA from a packed u32 (little-endian: R in low byte)
fn unpack_rgba(packed: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(packed & 0xFFu),
        f32((packed >> 8u) & 0xFFu),
        f32((packed >> 16u) & 0xFFu),
        f32((packed >> 24u) & 0xFFu)
    );
}

// Convert RGB float [0,255] to RGB565 u16
fn rgb_to_565(r: f32, g: f32, b: f32) -> u32 {
    let r5 = u32(clamp(r, 0.0, 255.0)) >> 3u;
    let g6 = u32(clamp(g, 0.0, 255.0)) >> 2u;
    let b5 = u32(clamp(b, 0.0, 255.0)) >> 3u;
    return (r5 << 11u) | (g6 << 5u) | b5;
}

// Expand RGB565 back to RGB [0,255] for palette computation
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

// Squared distance between two RGB vectors
fn color_dist_sq(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let d = a - b;
    return dot(d, d);
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let block_x = gid.x;
    let block_y = gid.y;

    if block_x >= params.blocks_x || block_y >= params.blocks_y {
        return;
    }

    // Load 16 pixels for this 4x4 block
    var pixels: array<vec3<f32>, 16>;
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
            let rgb = rgba.xyz;
            let pi = y * 4u + x;
            pixels[pi] = rgb;
            min_color = min(min_color, rgb);
            max_color = max(max_color, rgb);
        }
    }

    // Inset bounding box by 1/16 to reduce error
    let inset = (max_color - min_color) / 16.0;
    min_color = min_color + inset;
    max_color = max_color - inset;
    min_color = clamp(min_color, vec3<f32>(0.0), vec3<f32>(255.0));
    max_color = clamp(max_color, vec3<f32>(0.0), vec3<f32>(255.0));

    // Convert endpoints to RGB565
    var color0 = rgb_to_565(max_color.x, max_color.y, max_color.z);
    var color1 = rgb_to_565(min_color.x, min_color.y, min_color.z);

    // Ensure color0 > color1 for 4-color mode (no transparency)
    if color0 < color1 {
        let tmp = color0;
        color0 = color1;
        color1 = tmp;
    }
    if color0 == color1 {
        // Degenerate block - all same color, ensure 4-color mode
        if color0 > 0u {
            color1 = color0 - 1u;
        } else {
            color0 = 1u;
        }
    }

    // Build 4-color palette
    let p0 = rgb565_to_rgb(color0);
    let p1 = rgb565_to_rgb(color1);
    let p2 = (2.0 * p0 + p1) / 3.0;
    let p3 = (p0 + 2.0 * p1) / 3.0;

    // Encode each pixel as 2-bit index
    var indices = 0u;
    for (var i = 0u; i < 16u; i = i + 1u) {
        let px = pixels[i];
        let d0 = color_dist_sq(px, p0);
        let d1 = color_dist_sq(px, p1);
        let d2 = color_dist_sq(px, p2);
        let d3 = color_dist_sq(px, p3);

        var best_idx = 0u;
        var best_dist = d0;
        if d1 < best_dist { best_idx = 1u; best_dist = d1; }
        if d2 < best_dist { best_idx = 2u; best_dist = d2; }
        if d3 < best_dist { best_idx = 3u; }

        indices = indices | (best_idx << (i * 2u));
    }

    // Write output: 2 u32 values = 8 bytes per block
    let block_idx = (block_y * params.blocks_x + block_x) * 2u;
    output_blocks[block_idx] = color0 | (color1 << 16u);
    output_blocks[block_idx + 1u] = indices;
}
