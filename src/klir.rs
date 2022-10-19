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
struct Func {}

#[derive(Debug, Clone, Copy)]
struct FuncId {
    id: usize,
}

#[derive(Debug)]
enum Op {
    Add { lhs: VarId, rhs: VarId, dst: VarId },
    Sub { lhs: VarId, rhs: VarId, dst: VarId },
    Zero { dst: VarId },
    One { dst: VarId },
    FuncStart { args: Vec<VarId>, func: FuncId },
    FuncEnd { func: FuncId },
    FuncExec { func: FuncId, args: Vec<VarId> },
}

#[derive(Debug, Default)]
pub struct Ir {
    vars: Vec<Var>,
    ops: Vec<Op>,
    funcs: Vec<Func>,
}

impl Ir {
    fn alloc_var(&mut self) -> VarId {
        let dst = VarId {
            id: self.vars.len(),
        };
        self.vars.push(Var { ty: None });
        dst
    }
    fn new_func(&mut self) -> FuncId {
        let dst = FuncId {
            id: self.funcs.len(),
        };
        self.funcs.push(Func {});
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

    pub fn sub(&mut self, lhs: VarId, rhs: VarId) -> VarId {
        let dst = self.alloc_var();
        self.ops.push(Op::Sub { lhs, rhs, dst });
        dst
    }

    pub fn func<F>(&mut self, num_args: usize, f: F) -> FuncId
    where
        F: Fn(Vec<VarId>),
    {
        let func = self.new_func();
        let args = (0..num_args)
            .into_iter()
            .map(|_| self.alloc_var())
            .collect::<Vec<_>>();

        self.ops.push(Op::FuncStart {
            args: args.clone(),
            func,
        });

        f(args);

        self.ops.push(Op::FuncEnd { func });

        func
    }

    pub fn exec(&mut self, func: FuncId, args: &[VarId]) {
        self.ops.push(Op::FuncExec {
            func,
            args: args.to_vec(),
        });
    }
}
