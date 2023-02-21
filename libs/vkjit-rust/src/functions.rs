use vkjit_core::VarId;

use crate::*;

pub fn arange<T: From<VarId> + ToVarType>(num: usize) -> T {
    T::from(IR.lock().unwrap().arange(T::ty(), num))
}

pub fn linspace<T: Var + From<VarId> + ToVarType>(
    start: impl Into<T>,
    stop: impl Into<T>,
    num: usize,
) -> T {
    let start = start.into();
    let stop = stop.into();
    T::from(
        IR.lock()
            .unwrap()
            .linspace(T::ty(), start.id(), stop.id(), num),
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
