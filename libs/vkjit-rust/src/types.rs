use crate::IR;
use paste::paste;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use vkjit_core::internal::VarType;

use vkjit_core::{Ir, VarId};

pub trait Var {
    fn eval(self) -> Self;
    fn id(self) -> VarId;
    fn from_id(id: VarId) -> Self;
    fn ty() -> VarType;
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
            fn eval(self) -> Self {
                let res = IR.lock().unwrap().eval(vec![self.0]);
                Self(res[0])
            }
            fn id(self) -> VarId {
                self.0
            }
            fn from_id(id: VarId) -> Self {
                Self(id)
            }
            fn ty() -> VarType {
                VarType::$ty
            }
        }
    };
}

macro_rules! rs_bop {
    ($ty:ident, $op:ident) => {
        paste! {
            impl<Rhs: Into<Self>> $op<Rhs> for $ty {
                type Output = $ty;

                fn [<$op:lower>](self, rhs: Rhs) -> Self::Output{
                    let rhs = rhs.into();
                    Self(IR.lock().unwrap().[<$op:lower>](self.0, rhs.0))
                }
            }

            impl<Rhs: Into<Self>> [<$op Assign>]<Rhs> for $ty{
                fn [<$op:lower _assign>](&mut self, rhs: Rhs){
                    let rhs = rhs.into();
                    *self = self.[<$op:lower>](rhs);
                }
            }
        }
    };
}

macro_rules! bop {
    ($ty:ident, $op:ident) => {
        paste! {
            impl $ty{
                pub fn [<$op:lower>](self, rhs: impl Into<Self>) -> Bool{
                    let rhs = rhs.into();
                    Bool(IR.lock().unwrap().[<$op:lower>](self.0, rhs.0))
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

macro_rules! from_const {
    ($ty:ident) => {
        paste! {
            impl From<[<$ty:lower>]> for $ty {
                fn from(value: [<$ty:lower>]) -> Self {
                    Self(IR.lock().unwrap().[<const_ $ty:lower>](value))
                }
            }
        }
    };
}

macro_rules! from {
    ($ty:ident) => {
        paste! {
            impl From<&[[<$ty:lower>]]> for $ty {
                fn from(value: &[[<$ty:lower>]]) -> Self {
                    Self(IR.lock().unwrap().[<array_ $ty:lower>](value))
                }
            }
            impl<const N: usize> From<[[<$ty:lower>]; N]> for $ty{
                fn from(value: [[<$ty:lower>]; N]) -> Self {
                    Self(IR.lock().unwrap().[<array_ $ty:lower>](&value))
                }
            }
        }
    };
}

macro_rules! select {
    ($ty:ident) => {
        impl $ty {
            pub fn select(self, other: impl Into<Self>, condition: impl Into<Bool>) -> Self {
                let other = other.into();
                let condition = condition.into();
                Self(IR.lock().unwrap().select(condition.0, other.0, self.0))
            }
        }
    };
}

macro_rules! scatter {
    ($ty:ident) => {
        impl $ty {
            pub fn scatter(self, dst: Self, idx: impl Into<U32>) {
                let idx = idx.into();
                IR.lock().unwrap().scatter(self.0, dst.0, idx.0, None);
            }
            pub fn scatter_with(self, dst: Self, idx: impl Into<U32>, condition: impl Into<Bool>) {
                let idx = idx.into();
                let condition = condition.into();
                IR.lock()
                    .unwrap()
                    .scatter(self.0, dst.0, idx.0, Some(condition.0));
            }
        }
    };
}

var!(F32);

rs_bop!(F32, Add);
rs_bop!(F32, Sub);
rs_bop!(F32, Mul);
rs_bop!(F32, Div);

bop!(F32, Lt);
bop!(F32, Gt);
bop!(F32, Eq);
bop!(F32, Leq);
bop!(F32, Geq);
bop!(F32, Neq);

dbg!(F32);
from!(F32);
from_const!(F32);
select!(F32);
scatter!(F32);

var!(U32);

rs_bop!(U32, Add);
rs_bop!(U32, Sub);
rs_bop!(U32, Mul);
rs_bop!(U32, Div);

bop!(U32, Lt);
bop!(U32, Gt);
bop!(U32, Eq);
bop!(U32, Leq);
bop!(U32, Geq);
bop!(U32, Neq);

dbg!(U32);
from!(U32);
from_const!(U32);
select!(U32);
scatter!(U32);

var!(I32);

rs_bop!(I32, Add);
rs_bop!(I32, Sub);
rs_bop!(I32, Mul);
rs_bop!(I32, Div);

bop!(I32, Lt);
bop!(I32, Gt);
bop!(I32, Eq);
bop!(I32, Leq);
bop!(I32, Geq);
bop!(I32, Neq);

dbg!(I32);
from!(I32);
from_const!(I32);
select!(I32);
scatter!(I32);

var!(Bool);

from_const!(Bool);
select!(Bool);
scatter!(Bool);
