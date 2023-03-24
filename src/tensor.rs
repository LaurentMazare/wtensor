use crate::{Device, Error, Result};
use wgpu::util::DeviceExt;

pub trait Shape {
    fn num_dims(&self) -> usize;
    fn num_elements(&self) -> usize;
    fn dims(&self) -> Vec<usize>;
}

pub type D2 = (usize, usize);

impl Shape for (usize, usize) {
    fn num_dims(&self) -> usize {
        2
    }

    fn num_elements(&self) -> usize {
        self.0 * self.1
    }

    fn dims(&self) -> Vec<usize> {
        vec![self.0, self.1]
    }
}

pub trait Kind {
    fn size_of() -> usize;
}

impl Kind for f32 {
    fn size_of() -> usize {
        std::mem::size_of::<Self>()
    }
}

#[allow(dead_code)]
// TODO: Maybe we could encode in the type the device where the tensor lives?
/// A simple WebGPU backed tensor.
pub struct Tensor<S: Shape, K: Kind> {
    pub(crate) shape: S,
    pub(crate) data: wgpu::Buffer,
    pub(crate) device: Device,
    pub(crate) phantom: std::marker::PhantomData<K>,
}

impl Tensor<D2, f32> {
    fn new_uninitialized(
        device: &Device,
        width: usize,
        height: usize,
        mapped_at_creation: bool,
    ) -> Self {
        let dev = &device.0;
        let data = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (width * height * f32::size_of()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation,
        });
        Tensor {
            shape: (width, height),
            data,
            phantom: std::marker::PhantomData,
            device: device.clone(),
        }
    }

    pub fn new(device: &Device, width: usize, height: usize, v: f32) -> Self {
        let mut tensor = Self::new_uninitialized(device, width, height, true);
        tensor.fill(v);
        tensor
    }

    pub fn fill(&mut self, v: f32) {
        // TODO: Use a shader rather than transfering the data around.
        let data = vec![v; self.shape.num_elements()];
        self.data
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&data));
        self.data.unmap()
    }

    // TODO: Ensure that the same devices are used in self and rhs.
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        let (m, k1) = self.shape;
        let (k2, n) = rhs.shape;
        if k1 != k2 {
            return Err(Error::DimensionMismatchBinaryOp {
                op: "gemm",
                lhs: self.shape.dims(),
                rhs: rhs.shape.dims(),
            });
        }
        let dev = &self.device.0;
        let output = Self::new_uninitialized(&self.device, m, n, false);
        let param_buffer = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[m as u32, k1 as u32, n as u32]),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

        let bind_group = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &dev.mm_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rhs.data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: param_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut c = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            c.set_pipeline(&dev.mm_pipeline);
            c.set_bind_group(0, &bind_group, &[]);
            c.insert_debug_marker("gemm");
            c.dispatch_workgroups(m as u32, n as u32, 1);
        };
        dev.queue.submit(Some(encoder.finish()));
        Ok(output)
    }

    pub async fn to_vec(&self) -> Result<Vec<f32>> {
        let dev = &self.device.0;
        let size = self.shape.num_elements() as u64;
        let staging_buffer = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.data, 0, &staging_buffer, 0, size);
        self.device.0.queue.submit(Some(encoder.finish()));
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        dev.device.poll(wgpu::Maintain::Wait);

        receiver
            .receive()
            .await
            .ok_or(Error::ReceiverReturnedNone)??;
        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();
        // Since contents are got in bytes, this converts these bytes back to f32
        let result = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();
        Ok(result)
    }
}

pub type Tensor2D<K> = Tensor<D2, K>;
