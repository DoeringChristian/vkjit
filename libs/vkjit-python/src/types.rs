use paste::paste;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use vkjit_core::VarId;

use crate::IR;

pub trait Var {
    fn id(&self) -> VarId;
}

#[pyclass]
#[derive(Clone, Copy)]
pub struct F32(VarId);
#[pyclass]
#[derive(Clone, Copy)]
pub struct U32(VarId);
#[pyclass]
#[derive(Clone, Copy)]
pub struct I32(VarId);
#[pyclass]
#[derive(Clone, Copy)]
pub struct Bool(VarId);

macro_rules! new {
    ($wrapper:ident, $rtype:ident) => {
        paste! {
            #[pymethods]
            impl $wrapper {
                #[new]
                pub fn __new__(src: &PyAny) -> PyResult<Self> {
                    if let Ok(val) = src.extract::<f32>() {
                        return Ok(Self(IR.lock().unwrap().[<const_$rtype>](val)));
                    }
                    if let Ok(val) = src.extract::<Vec<f32>>() {
                        return Ok(Self(IR.lock().unwrap().[<array_$rtype>](&val)));
                    }
                    Err(PyTypeError::new_err("Not a valid argument!"))
                }
            }
        }
    };
}

new!(F32, f32);
