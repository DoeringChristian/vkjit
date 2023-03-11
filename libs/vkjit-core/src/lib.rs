#[allow(dead_code)]
mod backend;
#[allow(dead_code)]
pub mod internal;
mod iterators;
pub mod spv;
pub mod vartype;

mod test;

pub use internal::{Ir, VarId};
pub use vartype::{AsVarType, VarType};
