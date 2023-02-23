use vkjit_core::{AsVarType, VarId};

use crate::*;

pub fn arange<T: Into<Var> + AsVarType>(num: usize) -> Var {
    Var::from(IR.lock().unwrap().arange(T::as_var_type(), num))
}

pub fn linspace(start: impl Into<Var>, stop: impl Into<Var>, num: usize) -> Var {
    let start = start.into();
    let stop = stop.into();
    Var::from(
        IR.lock()
            .unwrap()
            .linspace(start.ty(), start.id(), stop.id(), num),
    )
}

#[macro_export]
macro_rules! eval {
    ($($var:expr)*) => {
        let schedule = [$($var.id()),*];
        $crate::eval_internal(&schedule);
    };
}

pub fn eval_internal(schedule: &[VarId]) {
    let mut ir = IR.lock().unwrap();
    ir.eval(&schedule);
}
