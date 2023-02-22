use paste::paste;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use vkjit_core::VarId;

use crate::IR;

macro_rules! from_const {
    ($ty:ident) => {
        paste! {
            impl From<$ty> for Var {
                fn from(value: $ty) -> Self {
                    let mut ir = IR.lock().unwrap();
                    Self(ir.[<const_$ty>](value))
                }
            }
        }
    };
}
macro_rules! from_slice {
    ($ty:ident) => {
        paste! {
            impl From<&[$ty]> for Var {
                fn from(value: &[$ty]) -> Self {
                    let mut ir = IR.lock().unwrap();
                    Self(ir.[<array_$ty>](value))
                }
            }
            impl From<Vec<$ty>> for Var {
                fn from(value: Vec<$ty>) -> Self {
                    let mut ir = IR.lock().unwrap();
                    Self(ir.[<array_$ty>](&value))
                }
            }
        }
    };
}

from_const!(f32);
from_const!(i32);
from_const!(u32);
from_const!(bool);

from_slice!(f32);
from_slice!(i32);
from_slice!(u32);

impl TryFrom<&PyAny> for Var {
    type Error = PyErr;

    fn try_from(value: &PyAny) -> Result<Self, Self::Error> {
        if let Ok(val) = value.extract::<Var>() {
            return Ok(val);
        }
        if let Ok(val) = value.extract::<i32>() {
            return Ok(val.into());
        }
        if let Ok(val) = value.extract::<f32>() {
            return Ok(val.into());
        }
        if let Ok(val) = value.extract::<bool>() {
            return Ok(val.into());
        }
        if let Ok(val) = value.extract::<Vec<u32>>() {
            return Ok(val.into());
        }
        if let Ok(val) = value.extract::<Vec<i32>>() {
            return Ok(val.into());
        }
        if let Ok(val) = value.extract::<Vec<f32>>() {
            return Ok(val.into());
        }

        Err(PyTypeError::new_err("Not a valid argument!"))
    }
}

#[pyclass]
#[derive(Clone, Copy)]
pub struct Var(VarId);

#[pymethods]
impl Var {
    #[new]
    pub fn __new__(args: &PyAny) -> PyResult<Self> {
        Self::try_from(args)
    }
    pub fn id(&self) -> usize {
        self.0.get_id()
    }
    pub fn tolist(&self) -> Vec<f32> {
        Vec::from(IR.lock().unwrap().as_slice::<f32>(self.0))
    }
    pub fn __add__(&self, rhs: &PyAny) -> PyResult<Self> {
        let rhs = Var::try_from(rhs)?;
        Ok(Self(IR.lock().unwrap().add(self.0, rhs.0)))
    }
    pub fn __sub__(&self, rhs: &PyAny) -> PyResult<Self> {
        let rhs = Var::try_from(rhs)?;
        Ok(Self(IR.lock().unwrap().sub(self.0, rhs.0)))
    }
    pub fn __mul__(&self, rhs: &PyAny) -> PyResult<Self> {
        let rhs = Var::try_from(rhs)?;
        Ok(Self(IR.lock().unwrap().mul(self.0, rhs.0)))
    }
    pub fn __div__(&self, rhs: &PyAny) -> PyResult<Self> {
        let rhs = Var::try_from(rhs)?;
        Ok(Self(IR.lock().unwrap().div(self.0, rhs.0)))
    }
    pub fn __repr__(&self) -> PyResult<String> {
        let ir = IR.lock().unwrap();
        if ir.is_buffer(&self.0) {
            Ok(format! {"array(dtype = {:?}, {})", ir.var(self.0).ty(), ir.str(self.0)})
        } else {
            Ok(format!("{:?}", ir.var(self.0)))
        }
    }
    pub fn __str__(&self) -> PyResult<String> {
        let mut ir = IR.lock().unwrap();
        if !ir.is_buffer(&self.0) {
            ir.eval(&[self.0]);
        }
        Ok(format! {"{}", ir.str(self.0)})
    }
}
