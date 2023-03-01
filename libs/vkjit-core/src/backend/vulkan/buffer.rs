use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;
use std::sync::Arc;

use super::device::Device;

pub struct BufferInfo {
    pub size: vk::DeviceSize,
    pub usage: vk::BufferUsageFlags,
    pub alignment: vk::DeviceSize,
    pub mapable: bool,
}

pub struct Buffer {
    pub allocation: Option<Allocation>,
    pub info: BufferInfo,
    pub buffer: vk::Buffer,
    pub device: Arc<Device>,
}

impl Buffer {
    pub fn create(device: &Arc<Device>, info: impl Into<BufferInfo>) -> Self {
        let backend = device.clone();
        let info = info.into();

        let buffer_info = vk::BufferCreateInfo {
            size: info.size,
            usage: info.usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let buffer = unsafe { backend.device.create_buffer(&buffer_info, None).unwrap() };
        let mut requirements = unsafe { backend.device.get_buffer_memory_requirements(buffer) };
        requirements.alignment = requirements.alignment.max(info.alignment);

        let memory_location = if info.mapable {
            MemoryLocation::CpuToGpu
        } else {
            MemoryLocation::GpuOnly
        };

        let allocation = backend
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
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();

        unsafe {
            backend
                .device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .unwrap();
        }

        Self {
            allocation: Some(allocation),
            buffer,
            device: backend,
            info,
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
            self.device.device.destroy_buffer(self.buffer, None);
        }
    }
}
