use crate::{Error, Result};
use std::borrow::Cow;

#[allow(dead_code)]
pub(crate) struct DeviceInternal {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) mm_pipeline: wgpu::ComputePipeline,
    pub(crate) add_pipeline: wgpu::ComputePipeline,
    pub(crate) fill_pipeline: wgpu::ComputePipeline,
}

#[derive(Clone)]
pub struct Device(pub(crate) std::rc::Rc<DeviceInternal>);

impl Device {
    pub async fn new() -> Result<Self> {
        Self::new_with_instance(wgpu::Instance::default()).await
    }

    pub async fn new_with_instance(instance: wgpu::Instance) -> Result<Self> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok_or_else(|| Error::NoAdapter(instance))?;
        let info = adapter.get_info();
        println!("AdapterInfo: {info:?}");

        let device_descriptor = wgpu::DeviceDescriptor {
            label: None,
            features: wgpu::Features::empty(),
            limits: adapter.limits(),
        };

        let (device, queue) = adapter.request_device(&device_descriptor, None).await?;
        println!("Device: {device:?}");

        let get_module = |s| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(s)),
            })
        };
        let mm_module = get_module(include_str!("shaders_gemm.wgsl"));
        let add_module = get_module(include_str!("shaders_add.wgsl"));
        let fill_module = get_module(include_str!("shaders_fill.wgsl"));
        let mm_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &mm_module,
            entry_point: "gemm",
        });
        let add_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &add_module,
            entry_point: "add",
        });
        let fill_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &fill_module,
            entry_point: "fill",
        });

        let internal = DeviceInternal {
            device,
            queue,
            mm_pipeline,
            add_pipeline,
            fill_pipeline,
        };
        Ok(Self(std::rc::Rc::new(internal)))
    }

    pub(crate) fn storage_buffer<K: crate::tensor::Kind>(
        &self,
        size: usize,
        mapped_at_creation: bool,
    ) -> wgpu::Buffer {
        self.0.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (size * K::size_of()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation,
        })
    }

    pub(crate) fn transfer_buffer<K: crate::tensor::Kind>(&self, size: usize) -> wgpu::Buffer {
        self.0.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (size * K::size_of()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub(crate) fn create_bind_group(
        &self,
        pl: &wgpu::ComputePipeline,
        buffers: &[&wgpu::Buffer],
    ) -> wgpu::BindGroup {
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        self.0.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pl.get_bind_group_layout(0),
            entries: entries.as_slice(),
        })
    }
}
