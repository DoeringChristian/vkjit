use bytemuck::cast_slice;
use screen_13::prelude::*;
use spirq::ReflectConfig;
use std::collections::HashMap;
use std::sync::Arc;

use rspirv::binary::{Assemble, Disassemble};

use crate::array::Array;
use crate::ir::Access;

#[allow(dead_code)]
mod array;
#[allow(dead_code)]
mod ir;

fn main() {
    pretty_env_logger::init();

    let cfg = DriverConfig::new()
        .debug(true)
        .presentation(false)
        .sync_display(false)
        .build();
    let sc13 = EventLoop::new().debug(true).build().unwrap();
    //let device = Arc::new(Device::new(cfg).unwrap());

    let mut i = ir::Ir::new(&sc13.device);

    // Record kernel
    //let x = i.arange(ir::VarType::Float32, 10);
    let x = i.array_f32(&[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);
    //let z = i.add(x, y);
    //let x = i.arange(ir::VarType::Float32, 10);

    let mut k = ir::Kernel::new();
    let res = k.compile(&mut i, vec![x]);

    println!("{:#?}", i);
    println!("{:#?}", res);
    println!("{:#?}", k);

    let mut graph = RenderGraph::new();
    let mut pool = LazyPool::new(&sc13.device);

    k.execute(&i, &mut graph);

    graph.resolve().submit(&mut pool, 0).unwrap();

    println!("res={:#?}", res[0]);
    let var = i.var(res[0]);
    println!("{:#?}", var.array.as_ref().unwrap());
    let res = Buffer::mapped_slice(&var.array.as_ref().unwrap().buf);
    let res: &[f32] = cast_slice(res);
    println!("{:#?}", res);
}
