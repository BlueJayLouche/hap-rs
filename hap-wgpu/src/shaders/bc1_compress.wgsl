// BC1/DXT1 GPU Compression Shader
//
// Compresses RGBA pixels to BC1 format (8 bytes per 4x4 block).
// Each invocation processes one 4x4 block. The colour-block fit itself lives
// in _refit.wgsl, shared with BC3 and YCoCg-BC3.

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
            let idx = (px_y + y) * params.width + px_x + x;
            fit_pixels[y * 4u + x] = unpack_rgba(input_pixels[idx]).xyz;
        }
    }

    let fit = encode_color_block();

    // Write output: 2 u32 values = 8 bytes per block
    let block_idx = (block_y * params.blocks_x + block_x) * 2u;
    output_blocks[block_idx] = fit.color0 | (fit.color1 << 16u);
    output_blocks[block_idx + 1u] = fit.indices;
}
