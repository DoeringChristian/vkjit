pub struct Var {
    pub size: Option<usize>, // Size of the variable
    pub access: Vec<usize>,  // Opperations accessing the variable
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
        start: f32,
        end: f32,
        num: usize,
    },
}

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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test01() {}
}
