use std::sync::Arc;

mod buffer;
mod device;
mod ir;
mod kernel;

use ash::vk;
use buffer::{Buffer, BufferInfo};
use device::Device;

fn main() {
    let device = Arc::new(Device::create());
    let buffer = Arc::new(
        Buffer::create(
            &device,
            BufferInfo {
                size: 100,
                usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
                can_map: false,
            },
        )
        .unwrap(),
    );
}
