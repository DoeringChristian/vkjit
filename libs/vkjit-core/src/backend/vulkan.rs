use bytemuck::cast_slice;
use crevice::std140::{self, AsStd140};
use screen_13::prelude::*;
use std::sync::Arc;

use crate::internal::{Access, Binding, Kernel};
use crate::VarType;

use super::*;

pub struct VulkanBackend {
    device: Arc<Device>,
}

impl Array for Arc<Buffer> {
    fn device_address(&self) -> u64 {
        Buffer::device_address(self)
    }

    fn map(&self) -> &[u8] {
        Buffer::mapped_slice(&self)
    }
    fn size(&self) -> usize {
        self.info.size as usize
    }
}

impl Backend for VulkanBackend {
    type Array = Arc<Buffer>;
    type Device = Arc<Device>;

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn create() -> Self {
        let cfg = screen_13::prelude::DriverConfig::new().build();
        let device = Arc::new(screen_13::prelude::Device::new(cfg).unwrap());
        Self { device }
    }

    fn create_array_from_slice(&self, data: &[u8]) -> Self::Array {
        let count = data.len();
        let size = count;
        let buf = Arc::new(
            Buffer::create_from_slice(&self.device, vk::BufferUsageFlags::STORAGE_BUFFER, data)
                .unwrap(),
        );
        buf
    }

    fn execute(&self, kernel: Kernel, dst: &[Self::Array]) {
        trace!("Recording Render Graph...");
        let mut graph = RenderGraph::new();
        let mut pool = LazyPool::new(&self.device);

        let num = kernel.num.unwrap();
        let arrays = kernel.arrays.clone();

        let spv = kernel.assemble();
        let pipeline = Arc::new(
            ComputePipeline::create(
                &self.device(),
                screen_13::prelude::ComputePipelineInfo::default(),
                screen_13::prelude::Shader::new_compute(spv),
            )
            .unwrap(),
        );
        trace!("{:?}", pipeline);

        // Collect nodes and corresponding bindings
        trace!("Collecting Nodes and Bindings...");

        let nodes = arrays
            .iter()
            .chain(dst)
            .map(|arr| graph.bind_node(arr))
            .collect::<Vec<_>>();

        let mut pass = graph.begin_pass("Eval kernel").bind_pipeline(&pipeline);

        for node in nodes {
            pass = pass.write_node(node);
        }

        trace!("Recording Compute Pass of size ({}, 1, 1)...", num);
        pass.record_compute(move |compute, _| {
            compute.dispatch(num as u32, 1, 1);
        })
        .submit_pass();

        trace!("Resolving Graph...");
        graph.resolve().submit(&mut pool, 0).unwrap();

        trace!("Executing Computations...");
        unsafe { self.device().device_wait_idle().unwrap() };
    }

    fn create_array(&self, size: usize) -> Self::Array {
        Arc::new(
            Buffer::create(
                &self.device,
                BufferInfo::new_mappable(size as u64, vk::BufferUsageFlags::STORAGE_BUFFER),
            )
            .unwrap(),
        )
    }
}
impl VulkanBackend {}
