use spirv;

pub struct SpirvBuilder {
    version: (u8, u8),

    capabilities: Vec<u32>,
    extensions: Vec<u32>,
    imports: Vec<u32>,
    memory_model: Vec<u32>,
    entry_points: Vec<u32>,
    execution_modes: Vec<u32>,
    annoations: Vec<u32>,
    types: Vec<u32>,
    function_decl: Vec<u32>,
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
