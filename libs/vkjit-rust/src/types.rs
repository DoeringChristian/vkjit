use crate::IR;
use paste::paste;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use vkjit_core::{Ir, VarId};

pub trait Var {
    fn id(&self) -> &VarId;
    fn id_mut(&mut self) -> &mut VarId;
}

#[derive(Clone, Copy)]
pub struct F32(VarId);
#[derive(Clone, Copy)]
pub struct U32(VarId);
#[derive(Clone, Copy)]
pub struct I32(VarId);
#[derive(Clone, Copy)]
pub struct Bool(VarId);

macro_rules! var {
    ($ty:ident) => {
        impl Var for $ty {
            fn id(&self) -> &VarId {
                &self.0
            }
            fn id_mut(&mut self) -> &mut VarId {
                &mut self.0
            }
        }
    };
}

macro_rules! bop {
    ($ty:ident, $op:ident) => {
        paste! {
            impl $op for $ty {
                type Output = $ty;

                fn [<$op:lower>](self, rhs: Self) -> Self::Output{
                    Self(IR.lock().unwrap().[<$op:lower>](self.0, rhs.0))
                }
            }

            impl [<$op Assign>] for $ty{
                fn [<$op:lower _assign>](&mut self, rhs: Self){
                    *self = self.[<$op:lower>](rhs);
                }
            }
        }
    };
}

macro_rules! dbg {
    ($ty:ident) => {
        paste! {
            impl Debug for $ty {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    f.debug_list()
                        .entries(IR.lock().unwrap().as_slice::<[<$ty:lower>]>(self.0))
                        .finish()
                }
            }

        }
    };
}

macro_rules! from {
    ($ty:ident) => {
        paste! {
            impl From<[<$ty:lower>]> for $ty {
                fn from(value: [<$ty:lower>]) -> Self {
                    Self(IR.lock().unwrap().[<const_ $ty:lower>](value))
                }
            }
            impl From<&[[<$ty:lower>]]> for $ty {
                fn from(value: &[[<$ty:lower>]]) -> Self {
                    Self(IR.lock().unwrap().[<array_ $ty:lower>](value))
                }
            }
        }
    };
}

var!(F32);

from!(F32);

bop!(F32, Add);
bop!(F32, Sub);
bop!(F32, Mul);
bop!(F32, Div);

var!(U32);

from!(U32);

bop!(U32, Add);
bop!(U32, Sub);
bop!(U32, Mul);
bop!(U32, Div);

var!(I32);

from!(I32);

bop!(I32, Add);
bop!(I32, Sub);
bop!(I32, Mul);
bop!(I32, Div);

dbg!(F32);
dbg!(U32);
dbg!(I32);

pub fn eval_single<T: Var>(val: &mut T) {
    let res = IR.lock().unwrap().eval(vec![*val.id()]);
    *val.id_mut() = res[0];
}
