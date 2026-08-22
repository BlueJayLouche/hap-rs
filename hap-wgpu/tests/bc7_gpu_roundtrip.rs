//! GPU round-trip validation for BC7 encoders.
//!
//! Encodes a test image, uploads the result as a `Bc7RgbaUnorm` texture,
//! renders it through a shader into an RGBA8 target, and compares the GPU's
//! hardware BC7 decode against the source pixels. This validates bit packing
//! against a real spec-compliant decoder (copying a compressed texture back
//! directly would return the compressed bytes, so a render pass is required).
//!
//! Covers both the CPU encoder (`hap_qt::bc7`) and the GPU compute shader
//! (`hap_wgpu::GpuDxtCompressor` with `HapFormat::Hap7`).
//!
//! Skips gracefully when no BC-capable adapter is available.

use pollster::block_on;

const W: u32 = 256; // 64 blocks * 16 bytes = 1024 bytes/row (multiple of 256)
const H: u32 = 64;

const SHADER: &str = r#"
@group(0) @binding(0) var tex: texture_2d<f32>;

struct VOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
    var out: VOut;
    let x = f32(i32(i / 2u) * 4 - 1);
    let y = f32(i32(i % 2u) * 4 - 1);
    out.pos = vec4f(x, y, 0.0, 1.0);
    out.uv = vec2f((x + 1.0) / 2.0, (1.0 - y) / 2.0);
    return out;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
    let dim = vec2f(textureDimensions(tex));
    return textureLoad(tex, vec2i(in.uv * dim), 0);
}
"#;

fn test_image() -> Vec<u8> {
    let mut rgba = vec![0u8; (W * H * 4) as usize];
    for y in 0..H {
        for x in 0..W {
            let i = ((y * W + x) * 4) as usize;
            rgba[i] = (x % 256) as u8; // horizontal ramp
            rgba[i + 1] = ((y * 4) % 256) as u8; // vertical ramp
            rgba[i + 2] = (((x + y * 3) / 2) % 256) as u8; // diagonal
            rgba[i + 3] = if (x / 16 + y / 16) % 2 == 0 { 255 } else { 64 }; // alpha blocks
        }
    }
    rgba
}

fn make_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        ..Default::default()
    }))
    .ok()?;
    if !adapter.features().contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
        return None;
    }
    block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("bc7-test"),
        required_features: wgpu::Features::TEXTURE_COMPRESSION_BC,
        ..Default::default()
    }))
    .ok()
}

/// Upload BC7 blocks, render them through the hardware decoder, read back RGBA8.
fn gpu_decode_bc7(device: &wgpu::Device, queue: &wgpu::Queue, bc7: &[u8]) -> Vec<u8> {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bc7"),
        size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Bc7RgbaUnorm,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bc7,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some((W / 4) * 16),
            rows_per_image: Some(H / 4),
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("target"),
        size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("blit"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("blit-layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("blit-pl"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("blit"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    let view = texture.create_view(&Default::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("blit-bg"),
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&view),
        }],
    });

    let readback_row = W * 4;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (readback_row * H) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("render"),
    });
    {
        let target_view = target.create_view(&Default::default());
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("blit-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(readback_row),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
    queue.submit(std::iter::once(encoder.finish()));

    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().expect("map");
    let data = slice.get_mapped_range().expect("range");

    let mut out = vec![0u8; (W * H * 4) as usize];
    for y in 0..H {
        let src = &data[(y * readback_row) as usize..][..(W * 4) as usize];
        let dst = &mut out[(y * W * 4) as usize..][..(W * 4) as usize];
        dst.copy_from_slice(src);
    }
    out
}

/// Compare decoded output against source; assert encoder-grade quality.
fn assert_quality(name: &str, src: &[u8], dec: &[u8]) {
    let mut max_err = 0u32;
    let mut total_err = 0u64;
    for (a, b) in src.iter().zip(dec.iter()) {
        let err = (*a as u32).abs_diff(*b as u32);
        max_err = max_err.max(err);
        total_err += err as u64;
    }
    let mean_err = total_err as f64 / src.len() as f64;
    eprintln!("{name}: max_err={max_err} mean_err={mean_err:.3}");
    // Mode-6 quality: endpoints quantized to 8 bits effective, 4-bit indices.
    // A bit-packing bug would produce garbage (max err near 255).
    assert!(max_err <= 16, "{name}: max_err {max_err} — bit packing likely wrong");
    assert!(mean_err < 4.0, "{name}: mean_err {mean_err} too large");
}

#[test]
fn bc7_cpu_encoder_gpu_decode() {
    let Some((device, queue)) = make_device() else {
        eprintln!("no BC-capable adapter, skipping");
        return;
    };
    let rgba = test_image();
    let mut bc7 = vec![0u8; (W * H) as usize];
    hap_qt::bc7::compress_bc7_mode6(&rgba, W as usize, H as usize, hap_qt::DxtQuality::Balanced.refine_iters(), &mut bc7);
    let decoded = gpu_decode_bc7(&device, &queue, &bc7);
    assert_quality("cpu encoder -> gpu decode", &rgba, &decoded);
}

#[test]
fn bc7_gpu_shader_gpu_decode() {
    let Some((device, queue)) = make_device() else {
        eprintln!("no BC-capable adapter, skipping");
        return;
    };
    let rgba = test_image();
    let device = std::sync::Arc::new(device);
    let queue = std::sync::Arc::new(queue);
    let compressor = hap_wgpu::GpuDxtCompressor::new(
        std::sync::Arc::clone(&device),
        std::sync::Arc::clone(&queue),
        W,
        H,
    )
    .expect("compressor");
    let bc7 = compressor
        .compress(&rgba, hap_qt::HapFormat::Hap7, hap_qt::DxtQuality::Balanced)
        .expect("gpu compress");
    assert_eq!(bc7.len(), (W * H) as usize);
    let decoded = gpu_decode_bc7(&device, &queue, &bc7);
    assert_quality("gpu shader -> gpu decode", &rgba, &decoded);
}
