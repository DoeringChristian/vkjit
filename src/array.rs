use crevice::std140::{self, AsStd140};
use screen_13::prelude::*;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::ir::VarType;

#[derive(Debug)]
pub struct Array {
    pub buf: Arc<Buffer>,
    device: Arc<Device>,
    count: usize,
}

impl Array {
    pub fn create(
        device: &Arc<Device>,
        ty: &VarType,
        count: usize,
        usage: vk::BufferUsageFlags,
    ) -> Self {
        let size = ty.size() * count;
        let mut buf = Buffer::create(device, BufferInfo::new(size as u64, usage)).unwrap();
        Self {
            buf: Arc::new(buf),
            count,
            device: device.clone(),
        }
    }
    pub fn from_slice<T: AsStd140>(
        device: &Arc<Device>,
        data: &[T],
        usage: vk::BufferUsageFlags,
    ) -> Self {
        let count = data.len();
        let size = T::std140_size_static() * count;
        let buf = Arc::new({
            let mut buf =
                Buffer::create(device, BufferInfo::new_mappable(size as u64, usage)).unwrap();
            let mut writer = std140::Writer::new(Buffer::mapped_slice_mut(&mut buf));
            writer.write(data).unwrap();
            buf
        });
        Self {
            buf,
            count: data.len(),
            device: device.clone(),
        }
    }
    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }
}
