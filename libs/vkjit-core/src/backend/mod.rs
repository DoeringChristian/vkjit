use crevice::std140::AsStd140;

use crate::internal::{Binding, Kernel};
use crate::AsVarType;

pub mod vulkan;

pub trait Array {
    fn device_address(&self) -> u64;
    fn map(&self) -> &[u8];
    fn size(&self) -> usize;
}

pub trait Backend {
    type Array: Array;
    type Device;

    fn device(&self) -> &Self::Device;

    fn create() -> Self;
    fn create_array_from_slice(&self, data: &[u8]) -> Self::Array;
    fn create_array(&self, size: usize) -> Self::Array;

    fn execute(&self, kernel: Kernel, dst: &[Self::Array]);
}
