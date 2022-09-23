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
    ir_mg.vars.push(Var::Array {
        num: Some(num),
        access: vec![opidx],
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
