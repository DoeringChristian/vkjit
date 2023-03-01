use vkjit_core::Ir;
use vkjit_rust::*;

fn main() {
    pretty_env_logger::init();

    let x = arange(VarType::F32, 1e7 as usize);

    eval!(x);

    dbg!(x);
}
