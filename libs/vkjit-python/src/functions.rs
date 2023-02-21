use pyo3::prelude::*;
use pyo3::types::PyList;
use vkjit_core::VarId;

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
