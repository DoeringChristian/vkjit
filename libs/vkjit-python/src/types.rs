use paste::paste;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use vkjit_core::VarId;

use crate::IR;

pub trait Var {
    fn id(&self) -> VarId;
}

macro_rules! var {
    ($ty:ident) => {
        paste! {

            #[pyclass]
            #[derive(Clone, Copy)]
            pub struct [<$ty:upper>](VarId);

            #[pymethods]
            impl [<$ty:upper>]{
                #[new]
                pub fn __new__(src: &PyAny) -> PyResult<Self> {
                    if let Ok(val) = src.extract::<f32>() {
                        return Ok(Self(IR.lock().unwrap().[<const_$ty>](val)));
                    }
                    if let Ok(val) = src.extract::<Vec<f32>>() {
                        return Ok(Self(IR.lock().unwrap().[<array_$ty>](&val)));
                    }
                    Err(PyTypeError::new_err("Not a valid argument!"))
                }
            }
        }
    };
}

var!(f32);
//
// #[pymethods]
// impl F32 {}
