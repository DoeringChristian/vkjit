use vkjit_core::Ir;
use vkjit_rust::*;

fn main() {
    pretty_env_logger::init();

    let x = Var::from([1, 2, 3].as_slice());

    eval!(x);

    dbg!(x);
}
