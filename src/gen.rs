use std::sync::{Arc, Mutex};

use crate::ir::*;

#[derive(Clone, Debug)]
pub struct VarRef {
    ir: Arc<Mutex<Ir>>,
    idx: usize,
}

impl std::ops::Add for VarRef {
    type Output = VarRef;

    fn add(self, rhs: Self) -> Self::Output {
        add(&self, &rhs)
    }
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

pub fn add(lhs: &VarRef, rhs: &VarRef) -> VarRef {
    let ir = lhs.ir.clone();
    let mut ir_mg = ir.lock().unwrap();

    let varidx = ir_mg.vars.len();
    let opidx = ir_mg.ops.len();

    ir_mg.vars.push(Var {
        size: None,
        access: vec![Access::write(opidx)],
    });
    ir_mg.vars[lhs.idx].access.push(Access::read(opidx));
    ir_mg.vars[rhs.idx].access.push(Access::read(opidx));
    ir_mg.ops.push(Op::Add {
        dst: varidx,
        lhs: lhs.idx,
        rhs: rhs.idx,
    });

    VarRef {
        ir: ir.clone(),
        idx: varidx,
    }
}

pub fn eval(src: &VarRef) {
    let ir = src.ir.clone();
    let mut ir_mg = ir.lock().unwrap();

    let opidx = ir_mg.ops.len();
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_add() {
        let ir = Arc::new(Mutex::new(Ir::new()));

        let a = linespace(0., 10., 10, &ir);
        let b = linespace(0., 10., 10, &ir);
        let c = a + b;

        println!("{:#?}", ir);
        assert_eq!(
            ir.lock().unwrap().vars[0],
            Var {
                size: Some(10),
                access: vec![
                    Access {
                        idx: 0,
                        ty: AccessType::Write
                    },
                    Access {
                        idx: 2,
                        ty: AccessType::Read
                    }
                ]
            }
        );
    }
}
