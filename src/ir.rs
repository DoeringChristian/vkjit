pub enum Var {
    Array {
        size: Option<usize>, // Size of the variable
        access: Vec<usize>,  // Opperations accessing the variable
    },
}

pub enum Op {
    Add {
        src0: usize,
        src1: usize,
        dst: usize,
    },
    Eval {
        src: usize,
    },
    Linspace {
        dst: usize,
        size: usize,
    },
}

pub struct Ir {
    pub vars: Vec<Var>,
    pub ops: Vec<Op>,
}
