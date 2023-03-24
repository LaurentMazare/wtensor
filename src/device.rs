use crate::{Error, Result};
use std::borrow::Cow;

#[allow(dead_code)]
pub(crate) struct DeviceInternal {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) mm_pipeline: wgpu::ComputePipeline,
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

        let mut limits = wgpu::Limits::downlevel_defaults();

        // TODO: Make this configurable?
        limits.max_buffer_size = 1 << 31;
        limits.max_storage_buffer_binding_size = 1 << 30;
        let device_descriptor = wgpu::DeviceDescriptor {
            label: None,
            features: wgpu::Features::empty(),
            limits,
        };

        let (device, queue) = adapter.request_device(&device_descriptor, None).await?;
        let info = adapter.get_info();
        println!("Device: {device:?}");
        println!("AdapterInfo: {info:?}");

        let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders.wgsl"))),
        });
        let mm_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &cs_module,
            entry_point: "gemm",
        });

        let internal = DeviceInternal {
            device,
            queue,
            mm_pipeline,
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
}
