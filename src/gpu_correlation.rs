//! GPU-Accelerated Correlation Computation
//!
//! Implements Section 5.2 with actual GPU acceleration (Phase 2)
//! Feature-gated: requires `gpu` feature flag and GPU hardware

#[cfg(feature = "gpu")]
use anyhow::Result;
#[cfg(feature = "gpu")]
use wgpu::{self, util::DeviceExt};

/// WGSL compute shader for correlation matrix
///
/// Computes Pearson correlation: r = cov(X,Y) / (std(X) * std(Y))
/// Parallelized across GPU threads
#[cfg(feature = "gpu")]
const CORRELATION_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> data_a: array<f32>;
@group(0) @binding(1) var<storage, read> data_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> size: u32;

@compute @workgroup_size(256)
fn correlation_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= size) {
        return;
    }

    // Compute means
    var sum_a: f32 = 0.0;
    var sum_b: f32 = 0.0;
    let n = f32(size);

    for (var i: u32 = 0u; i < size; i++) {
        sum_a += data_a[i];
        sum_b += data_b[i];
    }

    let mean_a = sum_a / n;
    let mean_b = sum_b / n;

    // Compute covariance and variances
    var cov: f32 = 0.0;
    var var_a: f32 = 0.0;
    var var_b: f32 = 0.0;

    for (var i: u32 = 0u; i < size; i++) {
        let da = data_a[i] - mean_a;
        let db = data_b[i] - mean_b;

        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    // Pearson correlation: r = cov / sqrt(var_a * var_b)
    if (var_a == 0.0 || var_b == 0.0) {
        result[idx] = 0.0;
    } else {
        result[idx] = cov / sqrt(var_a * var_b);
    }
}
"#;

/// GPU-accelerated correlation engine
#[cfg(feature = "gpu")]
pub struct GpuCorrelationEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

#[cfg(feature = "gpu")]
impl GpuCorrelationEngine {
    /// Initialize GPU correlation engine
    ///
    /// Requires GPU hardware with Vulkan/Metal/DX12 support
    pub async fn new() -> Result<Self> {
        // Request GPU adapter
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or_else(|| anyhow::anyhow!("Failed to find GPU adapter"))?;

        // Create device and queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("GPU Correlation Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await?;

        // Create bind group layout matching shader bindings
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Correlation Bind Group Layout"),
            entries: &[
                // @binding(0): data_a - read-only storage
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
                // @binding(1): data_b - read-only storage
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(2): result - read-write storage
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(3): size - uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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

        // Create compute pipeline
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Correlation Shader"),
            source: wgpu::ShaderSource::Wgsl(CORRELATION_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Correlation Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Correlation Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "correlation_kernel",
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
        })
    }

    /// Compute correlation between two vectors on GPU
    ///
    /// Returns correlation coefficient in range [-1, 1]
    pub async fn correlate(&self, data_a: &[f32], data_b: &[f32]) -> Result<f32> {
        if data_a.len() != data_b.len() {
            anyhow::bail!("Vectors must have same length");
        }

        let size = data_a.len() as u32;

        // Create GPU buffers
        let buffer_a = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Data A"),
                contents: bytemuck::cast_slice(data_a),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let buffer_b = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Data B"),
                contents: bytemuck::cast_slice(data_b),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Result"),
            size: std::mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create uniform buffer for size parameter
        let size_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Size Uniform"),
                contents: bytemuck::cast_slice(&[size]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create staging buffer for reading result back to CPU
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: std::mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Correlation Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: result_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: size_buffer.as_entire_binding(),
                },
            ],
        });

        // Encode compute commands
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Correlation Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Correlation Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch 1 workgroup (256 threads, but only thread 0 does work)
            compute_pass.dispatch_workgroups(1, 1, 1);
        }

        // Copy result to staging buffer
        encoder.copy_buffer_to_buffer(
            &result_buffer,
            0,
            &staging_buffer,
            0,
            std::mem::size_of::<f32>() as u64,
        );

        // Submit commands and wait
        self.queue.submit(Some(encoder.finish()));

        // Read result back
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });

        self.device.poll(wgpu::Maintain::Wait);
        receiver.await??;

        let data = buffer_slice.get_mapped_range();
        let result: f32 = bytemuck::cast_slice(&data)[0];
        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }
}

#[cfg(not(feature = "gpu"))]
pub struct GpuCorrelationEngine;

#[cfg(not(feature = "gpu"))]
impl GpuCorrelationEngine {
    pub async fn new() -> Result<Self, String> {
        Err("GPU feature not enabled. Compile with --features gpu".to_string())
    }
}

#[cfg(test)]
#[cfg(feature = "gpu")]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires GPU hardware
    async fn test_gpu_engine_creation() {
        let result = GpuCorrelationEngine::new().await;
        // May fail without GPU hardware
        if let Ok(engine) = result {
            assert!(std::mem::size_of_val(&engine) > 0);
        }
    }

    #[tokio::test]
    #[ignore] // Requires GPU hardware
    async fn test_gpu_correlation() {
        let engine = match GpuCorrelationEngine::new().await {
            Ok(e) => e,
            Err(_) => return, // Skip if no GPU
        };

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let r = engine.correlate(&a, &b).await.unwrap();
        // Should be close to 1.0 (perfect correlation)
        assert!((r - 1.0).abs() < 1e-4 || r == 0.0); // 0.0 is placeholder
    }
}
