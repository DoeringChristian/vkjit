use paste::paste;
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
            }
        }
    };
}

var! {
    impl f32{
        bop{
            add;
            sub;
            mul;
            div;
        }
    }
}
var! {
    impl i32{
        bop{
            add;
            sub;
            mul;
            div;
        }
    }
}
