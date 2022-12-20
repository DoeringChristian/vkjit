use std::collections::HashMap;
use std::sync::Arc;

use rspirv::binary::Disassemble;

mod array;
#[allow(dead_code)]
mod ir;

fn build() -> Result<(), rspirv::dr::Error> {
    let mut b = rspirv::dr::Builder::new();

    b.set_version(1, 3);
    b.memory_model(
        rspirv::spirv::AddressingModel::Logical,
        rspirv::spirv::MemoryModel::Simple,
    );

    let void = b.type_void();
    let voidf = b.type_function(void, vec![void]);
    let x = b.begin_function(
        void,
        None,
        (rspirv::spirv::FunctionControl::DONT_INLINE | rspirv::spirv::FunctionControl::CONST),
        voidf,
    );
    b.begin_block(None)?;
    b.ret()?;
    b.end_function()?;
    println!("{}", b.module().disassemble());
    Ok(())
}

fn main() {
    let mut i = ir::Ir::default();
    let x = i.arange(ir::VarType::UInt32, 10);
    let y = i.const_u32(2);
    let z = i.add(x, y);

    let m = i.compile(vec![z]);
    println!("{}", m.disassemble());
}
