use crate::*;

pub fn arange<T: Var>(num: usize) -> T {
    T::from_id(IR.lock().unwrap().arange(T::ty(), num))
}

pub fn linspace<T: Var>(start: impl Into<T>, stop: impl Into<T>, num: usize) -> T {
    let start = start.into();
    let stop = stop.into();
    T::from_id(
        IR.lock()
            .unwrap()
            .linspace(T::ty(), start.id(), stop.id(), num),
    )
}
