use std::fmt::Debug;
use std::ops;

use paste::paste;
use vkjit_core::vartype::VarType;
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

macro_rules! bop {
    ($bop:ident) => {
        paste! {
            impl<T: Into<Var>> ops::$bop<T> for Var {
                type Output = Var;

                fn [<$bop:lower>](self, rhs: T) -> Self::Output {
                    let rhs = rhs.into();
                    Self(IR.lock().unwrap().[<$bop:lower>](self.0, rhs.0))
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

impl From<VarId> for Var {
    fn from(value: VarId) -> Self {
        Self(value)
    }
}

pub struct Var(VarId);

impl Debug for Var {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ir = IR.lock().unwrap();
        if ir.is_buffer(&self.0) {
            f.debug_tuple("Var")
                .field(&ir.str(self.0).as_str())
                .finish()
        } else {
            f.debug_tuple("Var").field(&ir.var(self.0)).finish()
        }
    }
}

impl Clone for Var {
    fn clone(&self) -> Self {
        IR.lock().unwrap().inc_ref_count(self.0);
        Self(self.0.clone())
    }
}

impl Drop for Var {
    fn drop(&mut self) {
        IR.lock().unwrap().dec_ref_count(self.0);
    }
}

impl Var {
    pub fn id(&self) -> VarId {
        self.0
    }
    pub fn ty(&self) -> VarType {
        IR.lock().unwrap().var(self.0).ty().clone()
    }
}

bop!(Add);
bop!(Sub);
bop!(Mul);
bop!(Div);
