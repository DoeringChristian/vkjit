use std::sync::Mutex;

use lazy_static::lazy_static;
use vkjit_core::Ir;

mod functions;
mod types;

lazy_static! {
    pub static ref IR: Mutex<Ir> = Mutex::new(Ir::new());
}

pub use functions::*;
pub use types::*;
pub use vkjit_core::vartype::VarType;
