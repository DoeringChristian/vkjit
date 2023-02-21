use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use vkjit_core::VarId;

use crate::IR;

macro_rules! method_bops {
    {
        $ty:ident,
        $bop:ident
    } => {
        paste!{
            pub fn [<__ $bop __>](&self, rhs: &Self) -> Self{
                Self(IR.lock().unwrap().$bop(self.0, rhs.0))
            }
        }
    };
}

macro_rules! var {
    {
        impl $ty:ident{
            bop{
                $($bop:ident;)*
            }
        }
    } => {
        paste!{
            #[pyclass]
            #[derive(Clone, Copy)]
            pub struct [<$ty:upper>](VarId);

            #[pymethods]
            impl [<$ty:upper>] {

                #[new]
                pub fn __new__(src: &PyAny) -> PyResult<Self> {
                    if let Ok(val) = src.extract::<$ty>() {
                        return Ok(Self(IR.lock().unwrap().[<const_$ty>](val)));
                    }
                    if let Ok(val) = src.extract::<Vec<$ty>>() {
                        return Ok(Self(IR.lock().unwrap().[<array_$ty>](&val)));
                    }
                    Err(PyTypeError::new_err("Not a valid argument!"))
                }

                pub fn id(&self) -> usize{
                    self.0.get_id()
                }

                pub fn tolist(&self) -> Vec<$ty>{
                    Vec::from(IR.lock().unwrap().as_slice::<$ty>(self.0))
                }

                // Add binary ops
                $(
                    pub fn [<__ $bop __>](&self, rhs: &Self) -> Self {
                        Self(IR.lock().unwrap().$bop(self.0, rhs.0))
                    }
                )*

                pub fn __set__(&self, cond: &Bool, val: &Self){
                    todo!()
                }
            }
        }
    };
}

#[pyclass]
#[derive(Clone, Copy)]
pub struct Var(VarId);
#[pymethods]
impl Var {
    #[new]
    pub fn __new__(src: &PyAny) -> PyResult<Self> {
        if let Ok(val) = src.extract::<f32>() {
            return Ok(Self(IR.lock().unwrap().const_f32(val)));
        }
        if let Ok(val) = src.extract::<Vec<f32>>() {
            return Ok(Self(IR.lock().unwrap().array_f32(&val)));
        }
        if let Ok(val) = src.extract::<i32>() {
            return Ok(Self(IR.lock().unwrap().const_i32(val)));
        }
        if let Ok(val) = src.extract::<Vec<u32>>() {
            return Ok(Self(IR.lock().unwrap().array_u32(&val)));
        }
        if let Ok(val) = src.extract::<bool>() {
            return Ok(Self(IR.lock().unwrap().const_bool(val)));
        }
        Err(PyTypeError::new_err("Not a valid argument!"))
    }
    pub fn id(&self) -> usize {
        self.0.get_id()
    }
    pub fn tolist(&self) -> Vec<f32> {
        Vec::from(IR.lock().unwrap().as_slice::<f32>(self.0))
    }
    pub fn __add__(&self, rhs: &Self) -> Self {
        Self(IR.lock().unwrap().add(self.0, rhs.0))
    }
    pub fn __sub__(&self, rhs: &Self) -> Self {
        Self(IR.lock().unwrap().sub(self.0, rhs.0))
    }
    pub fn __mul__(&self, rhs: &Self) -> Self {
        Self(IR.lock().unwrap().mul(self.0, rhs.0))
    }
    pub fn __div__(&self, rhs: &Self) -> Self {
        Self(IR.lock().unwrap().div(self.0, rhs.0))
    }
    pub fn __str__(&self) -> String {
        if !IR.lock().unwrap().is_buffer(&self.0) {
            IR.lock().unwrap().eval(&[self.0]);
        }
        format! {"{:?}[\n{}]", IR.lock().unwrap().var(self.0).ty(), IR.lock().unwrap().str(self.0)}
    }
}
