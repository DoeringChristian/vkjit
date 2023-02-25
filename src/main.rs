use vkjit_core::Ir;
use vkjit_rust::*;

fn main() {
    pretty_env_logger::init();

    let x = Var::from(vec![1, 2, 3]);
    let y = "abc";
    let c = Var::from(1);
    println!("test");
    let y = c + x;

    println!("{:#?}", IR.lock().unwrap());

    eval!(y);

    println!("{:?}", y);
}
