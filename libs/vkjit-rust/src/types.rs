use log::{error, trace, warn};
use std::fmt::Debug;
use std::ops;

use paste::paste;
use vkjit_core::vartype::VarType;
use vkjit_core::VarId;

use crate::{gather, IR};

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

                    let mut ir = IR.lock().unwrap();
                    Self(ir.[<$bop:lower>](self.0, rhs.0))
                }
            }

            impl<T: Into<Var>> ops::[<$bop Assign>]<T> for Var{
                fn [<$bop:lower _assign>](&mut self, rhs: T) {
                    let rhs = rhs.into();

                    let ret = {
                        // MutexGuard to IR needs to be droped before asignment to self,
                        // which results in call to drop for self and locking of another
                        // Guard.
                        let mut ir = IR.lock().unwrap();
                        Self(ir.[<$bop:lower>](self.0, rhs.0))
                    };

                    *self = ret;
                }
            }
        }
    };
}

macro_rules! named_bop {
    ($bop:ident) => {
        paste! {
            impl Var {
                pub fn [<$bop:lower>](&self, rhs: impl Into<Var>) -> Self{
                    let rhs = rhs.into();
                    let mut ir = IR.lock().unwrap();

                    Self(ir.[<$bop:lower>](self.0, rhs.0))
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

impl From<&[Var]> for Var {
    fn from(vars: &[Var]) -> Self {
        let vars = vars.iter().map(|var| var.0).collect::<Vec<_>>();
        let id = IR.lock().unwrap().struct_init(&vars);
        Self(id)
    }
}

impl From<VarId> for Var {
    fn from(value: VarId) -> Self {
        Self(value)
    }
}

pub struct Var(VarId);

impl Debug for Var {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ir = IR.lock().unwrap();
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
        trace!("Dropping {:?}", self.id());
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
    pub fn getattr(&self, idx: usize) -> Self {
        Self(IR.lock().unwrap().getattr(self.0, idx))
    }
    pub fn setattr(&mut self, var: impl Into<Var>, idx: usize) {
        let var = var.into();
        let ret = {
            let mut ir = IR.lock().unwrap();
            Self(ir.setattr(self.0, var.0, idx))
        };
        *self = ret
    }
    pub fn then_else(&self, then: impl Into<Var>, other: impl Into<Var>) -> Self {
        let then = then.into();
        let other = other.into();

        let ret = Var::from(IR.lock().unwrap().select(self.id(), then.id(), other.id()));
        drop(then);
        drop(other);
        ret
    }
    pub fn scatter(&mut self, to: Var, idx: impl Into<Var>) {
        self.scatter_with(to, idx, None)
    }
    pub fn scatter_with(
        &mut self,
        to: Var,
        idx: impl Into<Var>,
        condition: impl Into<Option<Var>>,
    ) {
        let idx = idx.into();
        let condition = condition.into();
        let cond = condition.as_ref().map(|condition| condition.id());

        log::trace!("Scatter from {:?} to {:?} with {:?}", self.0, to.id(), cond);

        let ret = {
            let mut ir = IR.lock().unwrap();
            Self(ir.scatter(self.0, to.0, idx.0, cond))
        };
        *self = ret;
    }
    pub fn get(&self, idx: impl Into<Var>) -> Var {
        gather(self.clone(), idx)
    }
    pub fn to_vec<T: bytemuck::Pod>(&self) -> Vec<T> {
        let ir = IR.lock().unwrap();
        Vec::from(ir.as_slice::<T>(self.0))
    }
}

bop!(Add);
bop!(Sub);
bop!(Mul);
bop!(Div);
named_bop!(Lt);
named_bop!(Gt);
named_bop!(Eq);
named_bop!(Leq);
named_bop!(Geq);
named_bop!(Neq);
