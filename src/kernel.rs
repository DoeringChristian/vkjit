use std::sync::Arc;

use crate::buffer::Buffer;

pub struct BufferBinding {
    buffer: Arc<Buffer>,
    set: usize,
    binding: usize,
}

pub struct KSource {
    external: Vec<Arc<Buffer>>,
}
