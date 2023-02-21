mod functions;
mod types;

use functions::*;
use types::*;

use pyo3::prelude::*;

use lazy_static::lazy_static;
use std::sync::Mutex;

use vkjit_core::Ir;

lazy_static! {
    pub static ref IR: Mutex<Ir> = Mutex::new(Ir::new());
}
/// A Python module implemented in Rust.
#[pymodule]
fn vkjit(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Var>()?;
    m.add_function(wrap_pyfunction!(eval, m)?)?;
    Ok(())
}
