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
    let x = i.arange(ir::VarType::Float32, 10);
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
    let mut pool = LazyPool::new(&sc13.device);

    let module = k.b.module();
    println!("{}", module.disassemble());

    let spv = module.assemble();

    let entry_points = ReflectConfig::new()
        .spv(spv)
        .ref_all_rscs(true)
        .combine_img_samplers(true)
        .gen_unique_names(true)
        .reflect()
        .unwrap();
    println!("{:#?}", entry_points);

    //k.execute(&i, &mut graph);

    graph.resolve().submit(&mut pool, 0).unwrap();
}
