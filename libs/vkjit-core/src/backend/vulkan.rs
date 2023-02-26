use bytemuck::cast_slice;
use crevice::std140::{self, AsStd140};
use screen_13::prelude::*;
use std::sync::Arc;

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
}
