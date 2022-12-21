use screen_13::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

use rspirv::binary::{Assemble, Disassemble};

use crate::array::Array;
use crate::ir::Access;

#[allow(dead_code)]
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
    let cfg = DriverConfig::new().build();
    let device = Arc::new(Device::new(cfg).unwrap());

    let mut i = ir::Ir::new(&device);
    let x = i.arange(ir::VarType::UInt32, 10);
    let y = i.array_f32(&[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);
    let z = i.add(x, y);

    let mut k = ir::Kernel::new();
    let res = k.compile(&mut i, vec![z]);

    println!("{:#?}", i);
    println!("{:#?}", res);
    println!("{:#?}", k);
    //let module = k.b.module();
    //println!("{}", module.disassemble());

    let mut graph = RenderGraph::new();

    k.execute(&i, &mut graph);
}
