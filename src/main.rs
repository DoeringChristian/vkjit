use vkjit_core::Ir;
use vkjit_rust::*;

fn main() {
    let x = F32::from([1., 2., 3.]);
    let y = x + x;

    let y = y.eval();

    println!("{:?}", y);
}
