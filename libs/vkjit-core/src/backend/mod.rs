use crevice::std140::AsStd140;

use crate::AsVarType;

mod vulkan;

pub trait Backend {
    type Array;

    fn create() -> Self;
    fn create_array<T: AsVarType + AsStd140>(&mut self, data: &[T]) -> Self::Array;
    fn map_array<'a, T: AsVarType + AsStd140 + bytemuck::Pod>(
        &mut self,
        id: &'a Self::Array,
    ) -> &'a [T];
}
