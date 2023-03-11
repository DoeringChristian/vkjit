use spirv;

pub trait Write {
    fn num(&self) -> usize;
    fn write(&self, dst: &mut Vec<u32>);
}

pub struct Block {}

pub struct FuncDef {
    parameters: Vec<u32>,
    blocks: Vec<Block>,
}

#[derive(Default)]
pub struct SpirvBuilder {
    version: (u8, u8),

    capabilities: Vec<u32>,
    extensions: Vec<u32>,
    imports: Vec<u32>,
    memory_model: Vec<u32>,
    entry_points: Vec<u32>,
    execution_mode: Vec<u32>,
    // debug: Vec<u32>,
    annoations: Vec<u32>,
    types: Vec<u32>,
    // function_decl: Vec<u32>,
    function_def: Vec<u32>,
}

impl SpirvBuilder {
    pub fn build(&self) -> Vec<u32> {
        let mut dst = vec![];

        let bound = 0;

        // TODO: get bound

        dst.push(spirv::MAGIC_NUMBER); // 0
        dst.push(((self.version.0 as u32) << 16) | ((self.version.1 as u32) << 8)); // 1
        dst.push(0); // 2
        dst.push(bound); // 3
        dst.push(0); // 4

        return dst;
    }
}
