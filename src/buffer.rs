use std::sync::Arc;

use ash::vk;
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc},
    MemoryLocation,
};

use crate::device::Device;

pub struct Buffer {
    allocation: Option<Allocation>,
    buffer: vk::Buffer,
    device: Arc<Device>,
    pub info: BufferInfo,
    pub name: Option<String>,
}

impl Buffer {
    pub fn create(device: &Arc<Device>, info: impl Into<BufferInfo>) -> Option<Self> {
        let info = info.into();

        let device = device.clone();

        let buffer_info = vk::BufferCreateInfo {
            size: info.size,
            usage: info.usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let buffer = unsafe { device.create_buffer(&buffer_info, None).ok()? };

        let mut requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        if info
            .usage
            .contains(vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR)
        {
            requirements.alignment = requirements.alignment.max(64);
        }

        let memory_location = if info.can_map {
            MemoryLocation::CpuToGpu
        } else {
            MemoryLocation::GpuOnly
        };

        let allocation = device
            .allocator
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                name: "buffer",
                requirements,
                location: memory_location,
                linear: true,
            })
            .ok()?;

        unsafe {
            device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .ok()?;
        }

        Some(Self {
            allocation: Some(allocation),
            buffer,
            device,
            info,
            name: None,
        })
    }
}

pub struct BufferInfo {
    pub size: vk::DeviceSize,
    pub usage: vk::BufferUsageFlags,
    pub can_map: bool,
}
impl Default for BufferInfo {
    fn default() -> Self {
        Self {
            size: 1,
            usage: vk::BufferUsageFlags::empty(),
            can_map: false,
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        self.device
            .allocator
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .free(self.allocation.take().unwrap())
            .unwrap();
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
        }
    }
}
