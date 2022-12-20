use crevice::std140::{self, AsStd140};
use screen_13::prelude::*;
use std::marker::PhantomData;
use std::sync::Arc;

#[derive(AsStd140)]
struct Metadata {
    count: u32,
}

#[derive(Debug)]
pub struct Array<T: AsStd140> {
    _ty: PhantomData<T>,
    pub buf: Arc<Buffer>,
    device: Arc<Device>,
    count: usize,
}

impl<T: AsStd140> Array<T> {
    pub fn create(device: &Arc<Device>, data: &[T], usage: vk::BufferUsageFlags) -> Self {
        let count = data.len();
        let size = Metadata::std140_size_static() + T::std140_size_static() * count;
        let buf = Arc::new({
            let mut buf =
                Buffer::create(device, BufferInfo::new_mappable(size as u64, usage)).unwrap();
            let mut writer = std140::Writer::new(Buffer::mapped_slice_mut(&mut buf));
            writer
                .write(&Metadata {
                    count: count as u32,
                })
                .unwrap();
            writer.write(data).unwrap();
            buf
        });
        Self {
            buf,
            count: data.len(),
            device: device.clone(),
            _ty: PhantomData,
        }
    }
    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }
    #[inline]
    pub fn size(&self) -> usize {
        Metadata::std140_size_static() + T::std140_size_static() * self.count
    }
}
