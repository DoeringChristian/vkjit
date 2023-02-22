use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use vkjit_core::VarId;

use crate::types::Var;
use crate::IR;

#[pyfunction]
pub fn eval(schedule: &PyList) {
    let schedule = schedule
        .iter()
        .map(|id| {
            VarId::from(
                id.call_method("id", (), None)
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
            )
        })
        .collect::<Vec<_>>();

    IR.lock().unwrap().eval(&schedule);
}

#[pyfunction]
#[pyo3(signature = (*args))]
pub fn var(args: &PyTuple) -> PyResult<Var> {
    if args.len() == 1 {
        Var::try_from(args.get_item(0)?)
    } else {
        Var::try_from(args.as_ref())
    }
}

#[pyfunction]
pub fn ir() -> String {
    format!("{:#?}", IR.lock().unwrap())
}

#[pyfunction]
pub fn linspace(start: &PyAny, stop: &PyAny, num: usize) -> PyResult<Var> {
    let start = Var::try_from(start)?;
    let stop = Var::try_from(stop)?;

    assert_eq!(start.ty(), stop.ty());
    let ty = start.ty();

    let mut ir = IR.lock().unwrap();

    Ok(Var::from(ir.linspace(ty, start.id(), stop.id(), num)))
}
