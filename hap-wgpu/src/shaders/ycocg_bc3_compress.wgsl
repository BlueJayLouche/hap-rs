// YCoCg-BC3 GPU Compression Shader (HapY / Hap Q format)
//
// Converts RGBA to YCoCg colour space, then compresses to BC3/DXT5.
// This is the primary format for VJ use - high quality RGB without alpha.
//
// Standard HAP "scaled YCoCg" (id Software YCoCg-DXT5):
//   colour block carries (Co, Cg, scale-indicator), alpha block carries Y.
// Mirrors the CPU encoder in hap-qt's frame_encoder.rs.

// Convert RGB to YCoCg
fn rgb_to_ycocg(r: f32, g: f32, b: f32) -> vec3<f32> {
    let ri = i32(r);
    let gi = i32(g);
    let bi = i32(b);
    let y  = f32((ri + 2 * gi + bi) / 4);
    let co = f32((ri - bi) / 2 + 128);
    let cg = f32((-ri + 2 * gi - bi) / 4 + 128);
    return vec3<f32>(y, co, cg);
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let block_x = gid.x;
    let block_y = gid.y;

    if block_x >= params.blocks_x || block_y >= params.blocks_y {
        return;
    }

    var co_raw: array<f32, 16>;
    var cg_raw: array<f32, 16>;

    let px_x = block_x * 4u;
    let px_y = block_y * 4u;

    // Pass 1: YCoCg (Co/Cg centered at 128) and max chroma deviation.
    // Y goes to the alpha block, so it lands in fit_scalars directly.
    var max_dev = 0.0;
    for (var y = 0u; y < 4u; y = y + 1u) {
        for (var x = 0u; x < 4u; x = x + 1u) {
            let rgba = unpack_rgba(input_pixels[(px_y + y) * params.width + px_x + x]);
            let ycocg = rgb_to_ycocg(rgba.x, rgba.y, rgba.z); // (Y, Co, Cg)
            let pi = y * 4u + x;
            fit_scalars[pi] = ycocg.x;
            co_raw[pi] = ycocg.y;
            cg_raw[pi] = ycocg.z;
            max_dev = max(max_dev, max(abs(ycocg.y - 128.0), abs(ycocg.z - 128.0)));
        }
    }

    // Per-block chroma scale in {1,2,4}; blue stores (scale-1)*8 so the decoder
    // recovers it. Scaling expands chroma into more BC3 bits.
    var scale = 1.0;
    if max_dev <= 31.0 { scale = 4.0; } else if max_dev <= 63.0 { scale = 2.0; }
    let blue = (scale - 1.0) * 8.0;

    // Pass 2: build the scaled colour block.
    for (var i = 0u; i < 16u; i = i + 1u) {
        let co_s = clamp((co_raw[i] - 128.0) * scale + 128.0, 0.0, 255.0);
        let cg_s = clamp((cg_raw[i] - 128.0) * scale + 128.0, 0.0, 255.0);
        fit_pixels[i] = vec3<f32>(co_s, cg_s, blue);
    }

    // Blue is the scale indicator, not colour: pin it so neither refit nor
    // polish can drift it. blue is one of 0/8/24, each exact in RGB565.
    fit_lock_b5 = i32(u32(blue) >> 3u);

    let alpha = encode_alpha_block_from_scalars();
    let color = encode_color_block();

    // Write output: 4 u32 values = 16 bytes per block
    let block_idx = (block_y * params.blocks_x + block_x) * 4u;
    output_blocks[block_idx] = alpha.words.x;
    output_blocks[block_idx + 1u] = alpha.words.y;
    output_blocks[block_idx + 2u] = color.color0 | (color.color1 << 16u);
    output_blocks[block_idx + 3u] = color.indices;
}
