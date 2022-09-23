use std::sync::{Arc, Mutex};

use crate::ir::*;

pub struct VarRef {
    ir: Arc<Mutex<Ir>>,
    idx: usize,
}

pub fn linespace(start: f32, end: f32, num: usize, ir: &Arc<Mutex<Ir>>) -> VarRef {
    let mut ir_mg = ir.lock().unwrap();
    let varidx = ir_mg.vars.len();
    let opidx = ir_mg.ops.len();
    ir_mg.vars.push(Var {
        size: Some(num),
        access: vec![Access::write(opidx)],
    });
    ir_mg.ops.push(Op::Linspace {
        dst: varidx,
        start,
        end,
        num,
    });
    VarRef {
        ir: ir.clone(),
        idx: varidx,
    }
}

pub fn add(src0: &VarRef, src1: &VarRef) -> VarRef {
    let ir = src0.ir.clone();
    let mut ir_mg = ir.lock().unwrap();

    let varidx = ir_mg.vars.len();
    let opidx = ir_mg.ops.len();

    ir_mg.vars.push(Var {
        size: None,
        access: vec![Access::write(opidx)],
    });
    ir_mg.vars[src0.idx].access.push(Access::read(opidx));
    ir_mg.vars[src1.idx].access.push(Access::read(opidx));
    ir_mg.ops.push(Op::Add {
        dst: varidx,
        src0: src0.idx,
        src1: src1.idx,
    });

    VarRef {
        ir: ir.clone(),
        idx: varidx,
    }
}
