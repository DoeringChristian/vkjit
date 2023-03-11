use vkjit_core::Ir;
use vkjit_rust::*;

fn main() {
    pretty_env_logger::init();

    let x = arange(VarType::U32, 10 as usize);

    eval!(x);

    dbg!(x);
}
