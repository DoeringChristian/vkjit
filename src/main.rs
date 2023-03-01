use vkjit_core::Ir;
use vkjit_rust::*;

fn main() {
    pretty_env_logger::init();

    let x = arange(VarType::U32, 10);
    let y = x.clone() * 2.;

    eval!(x, y);

    dbg!(x);
}
