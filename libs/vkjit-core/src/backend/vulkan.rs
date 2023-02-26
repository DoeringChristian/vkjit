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

impl Backend for VulkanBackend {
    type Array = Arc<Buffer>;

    fn create() -> Self {
        let cfg = screen_13::prelude::DriverConfig::new().build();
        let device = Arc::new(screen_13::prelude::Device::new(cfg).unwrap());
        Self { device }
    }

    fn create_array<T: AsVarType + AsStd140>(&mut self, data: &[T]) -> Self::Array {
        let count = data.len();
        let size = T::std140_size_static() * count;
        let buf = Arc::new({
            let mut buf = Buffer::create(
                &self.device,
                BufferInfo::new_mappable(size as u64, vk::BufferUsageFlags::STORAGE_BUFFER),
            )
            .unwrap();
            let mut writer = std140::Writer::new(Buffer::mapped_slice_mut(&mut buf));
            writer.write(data).unwrap();
            buf
        });
        buf
    }

    fn map_array<'a, T: AsVarType + AsStd140 + bytemuck::Pod>(
        &mut self,
        array: &'a Self::Array,
    ) -> &'a [T] {
        let slice = Buffer::mapped_slice(&array);
        cast_slice(slice)
    }
    fn execute(&self, kernel: &Kernel, arrays: &[(Binding, Self::Array)]) {
        trace!("Recording Render Graph...");
        let mut graph = RenderGraph::new();
        let mut pool = LazyPool::new(&self.device);

        let spv = kernel.assemble();
        let pipeline = Arc::new(
            ComputePipeline::create(
                &self.device,
                screen_13::prelude::ComputePipelineInfo::default(),
                screen_13::prelude::Shader::new_compute(spv),
            )
            .unwrap(),
        );

        // Collect nodes and corresponding bindings
        trace!("Collecting Nodes and Bindings...");
        let nodes = arrays
            .iter()
            .map(|(binding, arr)| (*binding, graph.bind_node(arr)))
            .collect::<Vec<_>>();

        let mut pass = graph.begin_pass("Eval kernel").bind_pipeline(&pipeline);
        for (binding, node) in nodes {
            match binding.access {
                Access::Read => {
                    trace!("Binding buffer to {:?}", binding);
                    pass = pass.read_descriptor((binding.set, binding.binding), node);
                }
                Access::Write => {
                    trace!("Binding buffer to {:?}", binding);
                    pass = pass.write_descriptor((binding.set, binding.binding), node);
                }
            }
        }
        let num = kernel.num();
        trace!("Recording Compute Pass of size ({}, 1, 1)...", num);
        pass.record_compute(move |compute, _| {
            compute.dispatch(num as u32, 1, 1);
        })
        .submit_pass();

        trace!("Resolving Graph...");
        graph.resolve().submit(&mut pool, 0).unwrap();

        trace!("Executing Computations...");
        unsafe { self.device.device_wait_idle().unwrap() };
    }
}
impl VulkanBackend {}
