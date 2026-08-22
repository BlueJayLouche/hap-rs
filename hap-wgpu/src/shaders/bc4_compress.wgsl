// BC4/RGTC1 GPU Compression Shader (HapA / Alpha-only format)
//
// Compresses single-channel data to BC4 format (8 bytes per 4x4 block).
// Extracts the alpha channel from RGBA input; the fit lives in _refit.wgsl.

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
            let packed = input_pixels[(px_y + y) * params.width + px_x + x];
            fit_scalars[y * 4u + x] = f32((packed >> 24u) & 0xFFu);
        }
    }

    let alpha = encode_alpha_block_from_scalars();

    // Write output: 2 u32 values = 8 bytes per block
    let block_idx = (block_y * params.blocks_x + block_x) * 2u;
    output_blocks[block_idx] = alpha.words.x;
    output_blocks[block_idx + 1u] = alpha.words.y;
}
