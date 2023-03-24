use crate::{Device, Error, Result};
use wgpu::util::DeviceExt;

pub trait Shape: Clone {
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

impl<S: Shape, K: Kind> Tensor<S, K> {
    fn new_uninitialized(device: &Device, shape: S, mapped_at_creation: bool) -> Self {
        let data = device.storage_buffer::<K>(shape.num_elements(), mapped_at_creation);
        Tensor {
            shape,
            data,
            phantom: std::marker::PhantomData,
            device: device.clone(),
        }
    }
}

impl Tensor<D2, f32> {
    pub fn new(device: &Device, width: usize, height: usize, v: f32) -> Self {
        let mut tensor = Self::new_uninitialized(device, (width, height), false);
        tensor.fill(v);
        tensor
    }

    pub fn fill(&mut self, v: f32) {
        let (m, n) = self.shape;
        let dev = &self.device.0;
        let v_buffer = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[v]),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });
        let bind_group = self
            .device
            .create_bind_group(&dev.fill_pipeline, &[&self.data, &v_buffer]);
        let mut encoder = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let workgroups = (m * n + 63) / 64;
        {
            let mut c = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            c.set_pipeline(&dev.fill_pipeline);
            c.set_bind_group(0, &bind_group, &[]);
            c.insert_debug_marker("fill");
            c.dispatch_workgroups(1024, ((workgroups + 1023) / 1024) as u32, 1);
        };
        dev.queue.submit(Some(encoder.finish()));
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
        let output = Self::new_uninitialized(&self.device, (m, n), false);
        let param_buffer = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[m as u32, k1 as u32, n as u32]),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

        let bind_group = self.device.create_bind_group(
            &dev.mm_pipeline,
            &[&self.data, &rhs.data, &output.data, &param_buffer],
        );
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

    pub fn add(&self, rhs: &Self) -> Result<Self> {
        if self.shape != rhs.shape {
            return Err(Error::DimensionMismatchBinaryOp {
                op: "add",
                lhs: self.shape.dims(),
                rhs: rhs.shape.dims(),
            });
        }
        let (m, n) = self.shape;
        let dev = &self.device.0;
        let output = Self::new_uninitialized(&self.device, (m, n), false);
        let bind_group = self
            .device
            .create_bind_group(&dev.add_pipeline, &[&self.data, &rhs.data, &output.data]);
        let mut encoder = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let workgroups = (m * n + 63) / 64;
        {
            let mut c = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            c.set_pipeline(&dev.add_pipeline);
            c.set_bind_group(0, &bind_group, &[]);
            c.insert_debug_marker("add");
            c.dispatch_workgroups(1024, ((workgroups + 1023) / 1024) as u32, 1);
        };
        dev.queue.submit(Some(encoder.finish()));
        Ok(output)
    }
}

impl<S: Shape, K: Kind + bytemuck::AnyBitPattern> Tensor<S, K> {
    pub async fn to_vec(&self) -> Result<Vec<K>> {
        let dev = &self.device.0;
        let size = self.shape.num_elements();
        let staging_buffer = self.device.transfer_buffer::<K>(size);
        let mut encoder = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(
            &self.data,
            0,
            &staging_buffer,
            0,
            (size * K::size_of()) as u64,
        );
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
        // Since contents are got in bytes, this converts these bytes back to K
        let result = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();
        Ok(result)
    }
}

impl<S: Shape, K: Kind> Clone for Tensor<S, K> {
    fn clone(&self) -> Self {
        let dev = &self.device.0;
        let output = Self::new_uninitialized(&self.device, self.shape.clone(), false);
        let mut encoder = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(
            &self.data,
            0,
            &output.data,
            0,
            (self.shape.num_elements() * K::size_of()) as u64,
        );
        dev.queue.submit(Some(encoder.finish()));
        output
    }
}

pub type Tensor2D<K> = Tensor<D2, K>;
