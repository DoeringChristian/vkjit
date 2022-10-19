//
//Kernel Level Intermediate Representation

#[derive(Debug)]
enum VarType {
    F32,
}

#[derive(Debug)]
struct Var {
    ty: Option<VarType>,
}

#[derive(Debug, Clone, Copy)]
pub struct VarId {
    id: usize,
}

#[derive(Debug)]
enum Op {
    Add { lhs: VarId, rhs: VarId, dst: VarId },
    Zero { dst: VarId },
    One { dst: VarId },
    FnStart { args: Vec<VarId> },
}

#[derive(Debug, Default)]
pub struct Ir {
    vars: Vec<Var>,
    ops: Vec<Op>,
}

impl Ir {
    fn alloc_var(&mut self) -> VarId {
        let dst = VarId {
            id: self.vars.len(),
        };
        self.vars.push(Var { ty: None });
        dst
    }

    pub fn zero(&mut self) -> VarId {
        let dst = self.alloc_var();
        self.ops.push(Op::Zero { dst });
        return dst;
    }

    pub fn one(&mut self) -> VarId {
        let dst = self.alloc_var();
        self.ops.push(Op::One { dst });
        return dst;
    }

    pub fn add(&mut self, lhs: VarId, rhs: VarId) -> VarId {
        let dst = self.alloc_var();
        self.ops.push(Op::Add { lhs, rhs, dst });
        dst
    }
}
