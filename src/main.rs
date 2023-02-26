use vkjit_core::Ir;
use vkjit_rust::*;

fn main() {
    pretty_env_logger::init();

    let x = Var::from(vec![1., 2., 3.]) + 1.;

    eval!(x);

    dbg!(x);
}
