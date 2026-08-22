// BC3/DXT5 GPU Compression Shader
//
// Compresses RGBA pixels to BC3 format (16 bytes per 4x4 block).
// BC3 = alpha block (8 bytes) + BC1 colour block (8 bytes); both fits live in
// _refit.wgsl. Each invocation processes one 4x4 block.

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let block_x = gid.x;
    let block_y = gid.y;

    if block_x >= params.blocks_x || block_y >= params.blocks_y {
        return;
    }

    let px_x = block_x * 4u;
    let px_y = block_y * 4u;
    for (var y = 0u; y < 4u; y = y + 1u) {
        for (var x = 0u; x < 4u; x = x + 1u) {
            let rgba = unpack_rgba(input_pixels[(px_y + y) * params.width + px_x + x]);
            let pi = y * 4u + x;
            fit_pixels[pi] = rgba.xyz;
            fit_scalars[pi] = rgba.w;
        }
    }

    let alpha = encode_alpha_block_from_scalars();
    let color = encode_color_block();

    // Write output: 4 u32 values = 16 bytes per block
    // Layout: [alpha_word0][alpha_word1][color0|color1][color_indices]
    let block_idx = (block_y * params.blocks_x + block_x) * 4u;
    output_blocks[block_idx] = alpha.words.x;
    output_blocks[block_idx + 1u] = alpha.words.y;
    output_blocks[block_idx + 2u] = color.color0 | (color.color1 << 16u);
    output_blocks[block_idx + 3u] = color.indices;
}
