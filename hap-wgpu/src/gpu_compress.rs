//! GPU-Accelerated DXT Compression
//!
//! Uses wgpu compute shaders to compress RGBA pixels to DXT/BCn formats.
//! Falls back gracefully to CPU compression when GPU is unavailable.

use hap_qt::HapFormat;
use std::sync::Arc;
use thiserror::Error;

/// Errors from GPU compression
#[derive(Error, Debug)]
pub enum GpuCompressError {
    #[error("GPU buffer mapping failed: {0}")]
    BufferMapFailed(String),

    #[error("Invalid input size: expected {expected} bytes, got {got}")]
    InvalidInputSize { expected: usize, got: usize },

    #[error("Unsupported format for GPU compression: {0:?}")]
    UnsupportedFormat(HapFormat),

    #[error("GPU device error: {0}")]
    DeviceError(String),
}

/// Parameters passed to compute shaders via uniform buffer
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CompressParams {
    width: u32,
    height: u32,
    blocks_x: u32,
    blocks_y: u32,
}

/// GPU DXT compressor using wgpu compute shaders
pub struct GpuDxtCompressor {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    // Compute pipelines for each supported format
    bc1_pipeline: wgpu::ComputePipeline,
    bc3_pipeline: wgpu::ComputePipeline,
    bc4_pipeline: wgpu::ComputePipeline,
    ycocg_bc3_pipeline: wgpu::ComputePipeline,

    // Shared bind group layout
    bind_group_layout: wgpu::BindGroupLayout,

    // Buffers
    input_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,

    // Dimensions
    width: u32,
    height: u32,
    blocks_x: u32,
    blocks_y: u32,
}

impl GpuDxtCompressor {
    /// Try to create a GPU compressor. Returns None if GPU is not suitable.
    pub fn try_new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        width: u32,
        height: u32,
    ) -> Option<Self> {
        match Self::new(device, queue, width, height) {
            Ok(compressor) => Some(compressor),
            Err(_) => None,
        }
    }

    /// Create a new GPU DXT compressor for the given dimensions
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        width: u32,
        height: u32,
    ) -> Result<Self, GpuCompressError> {
        if width == 0 || height == 0 {
            return Err(GpuCompressError::DeviceError(
                "Width and height must be > 0".to_string(),
            ));
        }

        // Pad to multiples of 4
        let padded_w = ((width + 3) / 4) * 4;
        let padded_h = ((height + 3) / 4) * 4;
        let blocks_x = padded_w / 4;
        let blocks_y = padded_h / 4;

        // Create bind group layout (shared by all pipelines)
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("dxt_compress_bind_group_layout"),
                entries: &[
                    // Input pixels (storage, read-only)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Output blocks (storage, read-write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Params (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dxt_compress_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipelines from embedded WGSL shaders
        let bc1_pipeline =
            Self::create_pipeline(&device, &pipeline_layout, "bc1", include_str!("shaders/bc1_compress.wgsl"));
        let bc3_pipeline =
            Self::create_pipeline(&device, &pipeline_layout, "bc3", include_str!("shaders/bc3_compress.wgsl"));
        let bc4_pipeline =
            Self::create_pipeline(&device, &pipeline_layout, "bc4", include_str!("shaders/bc4_compress.wgsl"));
        let ycocg_bc3_pipeline = Self::create_pipeline(
            &device,
            &pipeline_layout,
            "ycocg_bc3",
            include_str!("shaders/ycocg_bc3_compress.wgsl"),
        );

        // Allocate buffers
        let input_size = (padded_w * padded_h * 4) as u64;
        // Max output size is BC3/DXT5: 16 bytes per block
        let max_output_size = (blocks_x * blocks_y * 16) as u64;

        let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dxt_input_buffer"),
            size: input_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dxt_output_buffer"),
            size: max_output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dxt_readback_buffer"),
            size: max_output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dxt_params_buffer"),
            size: std::mem::size_of::<CompressParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Upload initial params
        let params = CompressParams {
            width: padded_w,
            height: padded_h,
            blocks_x,
            blocks_y,
        };
        queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        Ok(Self {
            device,
            queue,
            bc1_pipeline,
            bc3_pipeline,
            bc4_pipeline,
            ycocg_bc3_pipeline,
            bind_group_layout,
            input_buffer,
            output_buffer,
            readback_buffer,
            params_buffer,
            width: padded_w,
            height: padded_h,
            blocks_x,
            blocks_y,
        })
    }

    fn create_pipeline(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        label: &str,
        source: &str,
    ) -> wgpu::ComputePipeline {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{}_shader", label)),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{}_pipeline", label)),
            layout: Some(layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    }

    /// Compress RGBA pixel data to DXT format on the GPU
    ///
    /// # Arguments
    /// * `rgba_data` - RGBA pixel data (width * height * 4 bytes, padded to multiples of 4)
    /// * `format` - Target HAP format
    ///
    /// # Returns
    /// Raw DXT compressed data suitable for `HapFrameEncoder::encode_from_dxt()`
    pub fn compress(
        &self,
        rgba_data: &[u8],
        format: HapFormat,
    ) -> Result<Vec<u8>, GpuCompressError> {
        let expected_input = (self.width * self.height * 4) as usize;
        if rgba_data.len() != expected_input {
            return Err(GpuCompressError::InvalidInputSize {
                expected: expected_input,
                got: rgba_data.len(),
            });
        }

        let (pipeline, bytes_per_block) = match format {
            HapFormat::Hap1 => (&self.bc1_pipeline, 8u32),
            HapFormat::Hap5 => (&self.bc3_pipeline, 16u32),
            HapFormat::HapY => (&self.ycocg_bc3_pipeline, 16u32),
            HapFormat::HapA => (&self.bc4_pipeline, 8u32),
            _ => return Err(GpuCompressError::UnsupportedFormat(format)),
        };

        let output_size = (self.blocks_x * self.blocks_y * bytes_per_block) as u64;

        // Upload RGBA data to input buffer
        self.queue
            .write_buffer(&self.input_buffer, 0, rgba_data);

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dxt_compress_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch compute shader
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dxt_compress_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dxt_compress_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(self.blocks_x, self.blocks_y, 1);
        }

        // Copy output to readback buffer
        encoder.copy_buffer_to_buffer(&self.output_buffer, 0, &self.readback_buffer, 0, output_size);

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and read back the result
        let buffer_slice = self.readback_buffer.slice(..output_size);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        let _ = self.device.poll(wgpu::PollType::Wait);

        receiver
            .recv()
            .map_err(|e| GpuCompressError::BufferMapFailed(e.to_string()))?
            .map_err(|e| GpuCompressError::BufferMapFailed(format!("{:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let result = data.to_vec();
        drop(data);
        self.readback_buffer.unmap();

        Ok(result)
    }

    /// Get the supported formats for GPU compression
    pub fn supported_formats() -> &'static [HapFormat] {
        &[HapFormat::Hap1, HapFormat::Hap5, HapFormat::HapY, HapFormat::HapA]
    }

    /// Check if a format is supported for GPU compression
    pub fn supports_format(format: HapFormat) -> bool {
        matches!(format, HapFormat::Hap1 | HapFormat::Hap5 | HapFormat::HapY | HapFormat::HapA)
    }

    /// Get the configured dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_params_layout() {
        // Verify CompressParams is the right size for uniform buffer
        assert_eq!(std::mem::size_of::<CompressParams>(), 16);
    }

    #[test]
    fn test_supported_formats() {
        assert!(GpuDxtCompressor::supports_format(HapFormat::Hap1));
        assert!(GpuDxtCompressor::supports_format(HapFormat::HapY));
        assert!(!GpuDxtCompressor::supports_format(HapFormat::Hap7));
        assert!(!GpuDxtCompressor::supports_format(HapFormat::HapH));
    }
}
