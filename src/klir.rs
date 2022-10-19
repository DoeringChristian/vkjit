//
//Kernel Level Intermediate Representation

#[derive(Debug, Clone, Copy)]
enum VarType {
    F32,
}

#[derive(Debug, Clone)]
struct Var {
    ty: Option<VarType>,
}

#[derive(Debug, Clone, Copy)]
pub struct VarId {
    id: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct OpId {
    id: usize,
}

#[derive(Debug, Clone)]
enum Op {
    Add {
        lhs: VarId,
        rhs: VarId,
        dst: VarId,
    },
    Sub {
        lhs: VarId,
        rhs: VarId,
        dst: VarId,
    },
    Zero {
        dst: VarId,
    },
    One {
        dst: VarId,
    },
    FuncStart {
        args: Vec<VarId>,
        ret: Option<VarId>,
    },
    FuncEnd {},
    FuncExec {
        op: OpId,
        args: Vec<VarId>,
        ret: Option<VarId>,
    },
}

#[derive(Debug, Default)]
pub struct Ir {
    vars: Vec<Var>,
    ops: Vec<Op>,
}

impl Ir {
    fn new_var(&mut self) -> VarId {
        let dst = VarId {
            id: self.vars.len(),
        };
        self.vars.push(Var { ty: None });
        dst
    }
    fn push_op(&mut self, op: Op) -> OpId {
        let opid = OpId { id: self.ops.len() };
        self.ops.push(op);
        opid
    }

    fn var_mut(&mut self, var: VarId) -> &mut Var {
        &mut self.vars[var.id]
    }
    fn op_mut(&mut self, op: OpId) -> &mut Op {
        &mut self.ops[op.id]
    }
    fn get_op(&self, op: OpId) -> Op {
        self.ops[op.id].clone()
    }

    pub fn zero(&mut self) -> VarId {
        let dst = self.new_var();
        self.ops.push(Op::Zero { dst });
        return dst;
    }

    pub fn one(&mut self) -> VarId {
        let dst = self.new_var();
        self.ops.push(Op::One { dst });
        return dst;
    }

    pub fn add(&mut self, lhs: VarId, rhs: VarId) -> VarId {
        let dst = self.new_var();
        self.ops.push(Op::Add { lhs, rhs, dst });
        dst
    }

    pub fn sub(&mut self, lhs: VarId, rhs: VarId) -> VarId {
        let dst = self.new_var();
        self.ops.push(Op::Sub { lhs, rhs, dst });
        dst
    }

    pub fn func<F>(&mut self, num_args: usize, f: F) -> OpId
    where
        F: Fn(&mut Ir, Vec<VarId>) -> Option<VarId>,
    {
        let args = (0..num_args)
            .into_iter()
            .map(|_| self.new_var())
            .collect::<Vec<_>>();

        let op = self.push_op(Op::FuncStart {
            args: args.clone(),
            ret: None,
        });

        let _ret = f(self, args);

        match self.op_mut(op) {
            Op::FuncStart { ref mut ret, .. } => *ret = _ret,
            _ => unreachable!(),
        };
        self.ops.push(Op::FuncEnd {});

        op
    }

    pub fn exec(&mut self, op: OpId, args: &[VarId]) -> Option<VarId> {
        let ret = match self.get_op(op) {
            Op::FuncStart { mut args, mut ret } => ret.and_then(|_| Some(self.new_var())),
            _ => unreachable!(),
        };

        self.ops.push(Op::FuncExec {
            op,
            args: args.to_vec(),
            ret,
        });

        ret
    }
}
