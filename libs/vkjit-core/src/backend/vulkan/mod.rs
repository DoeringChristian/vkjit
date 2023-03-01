use std::ffi::CStr;
use std::sync::Arc;

use self::buffer::{Buffer, BufferInfo};
use self::device::Device;

use super::{Array, Backend};

use ash::vk;

mod buffer;
mod device;

pub struct VulkanBackend {
    device: Arc<Device>,
}

impl Array for Arc<Buffer> {
    fn device_address(&self) -> u64 {
        unsafe {
            self.device.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::builder().buffer(self.buffer),
            )
        }
    }

    fn map(&self) -> &[u8] {
        self.allocation.as_ref().unwrap().mapped_slice().unwrap()
    }

    fn size(&self) -> usize {
        self.info.size as _
    }
}

impl Backend for VulkanBackend {
    type Array = Arc<Buffer>;

    // type Device = Arc<Device>;

    // fn device(&self) -> &Self::Device {
    //     todo!()
    // }

    fn create() -> Self {
        Self {
            device: Arc::new(Device::create()),
        }
    }

    fn create_array_from_slice(&self, data: &[u8]) -> Self::Array {
        let mut buf = Buffer::create(
            &self.device,
            BufferInfo {
                size: data.len() as u64,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                alignment: 4,
                mapable: true,
            },
        );

        let _ = &mut buf.allocation.as_mut().unwrap().mapped_slice_mut().unwrap()
            [0..buf.info.size as usize]
            .copy_from_slice(&data);

        Arc::new(buf)
    }

    fn create_array(&self, size: usize) -> Self::Array {
        Arc::new(Buffer::create(
            &self.device,
            BufferInfo {
                size: size as u64,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                alignment: 4,
                mapable: true,
            },
        ))
    }

    fn execute(&self, kernel: crate::internal::Kernel, dst: &[Self::Array]) {
        let num = kernel.num.unwrap();
        let code = kernel.assemble();

        unsafe {
            let shader_info = vk::ShaderModuleCreateInfo::builder().code(&code).build();
            let shader = self
                .device
                .device
                .create_shader_module(&shader_info, None)
                .unwrap();

            let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&[]);

            let pipeline_layout = self
                .device
                .device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .unwrap();
            let pipeline_cache = self
                .device
                .device
                .create_pipeline_cache(&vk::PipelineCacheCreateInfo::builder(), None)
                .unwrap();

            let pipeline_shader_info = vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader)
                .name(CStr::from_bytes_with_nul(b"Eval\0").unwrap());
            let compute_pipeline_info = vk::ComputePipelineCreateInfo::builder()
                .stage(pipeline_shader_info.build())
                .layout(pipeline_layout);

            let compute_pipeline = self
                .device
                .device
                .create_compute_pipelines(pipeline_cache, &[compute_pipeline_info.build()], None)
                .unwrap()[0];

            let cmd_pool_info = vk::CommandPoolCreateInfo::builder()
                .queue_family_index(self.device.queue_family_index);

            let cmd_pool = self
                .device
                .device
                .create_command_pool(&cmd_pool_info, None)
                .unwrap();

            let cmd_buf_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            let cmd_buffer = self
                .device
                .device
                .allocate_command_buffers(&cmd_buf_info)
                .unwrap()[0];

            let begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            self.device
                .device
                .begin_command_buffer(cmd_buffer, &begin_info)
                .unwrap();

            self.device.device.cmd_bind_pipeline(
                cmd_buffer,
                vk::PipelineBindPoint::COMPUTE,
                compute_pipeline,
            );
            self.device.device.cmd_dispatch(cmd_buffer, num as _, 1, 1);

            self.device.device.device_wait_idle().unwrap();
        }
    }
}
