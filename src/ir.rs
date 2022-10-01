#[derive(Debug, PartialEq, Eq)]
pub enum AccessType {
    Read,
    Write,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Access {
    pub idx: usize,
    pub ty: AccessType,
}

impl Access {
    pub fn read(idx: usize) -> Self {
        Self {
            idx,
            ty: AccessType::Read,
        }
    }
    pub fn write(idx: usize) -> Self {
        Self {
            idx,
            ty: AccessType::Write,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct Var {
    pub size: Option<usize>, // Size of the variable
    pub access: Vec<Access>, // Opperations accessing the variable
}

#[derive(Debug)]
pub enum Op {
    Add {
        lhs: usize,
        rhs: usize,
        dst: usize,
    },
    Linspace {
        dst: usize,
        start: f32,
        end: f32,
        num: usize,
    },
}

#[derive(Debug)]
pub struct Ir {
    pub vars: Vec<Var>,
    pub ops: Vec<Op>,
}

impl Ir {
    pub fn new() -> Self {
        Self {
            vars: vec![],
            ops: vec![],
        }
    }
}
