#[allow(dead_code)]
mod array;
#[allow(dead_code)]
pub mod internal;
mod iterators;
pub mod vartype;

mod test;

pub use internal::{Ir, VarId};
pub use vartype::{AsVarType, VarType};
