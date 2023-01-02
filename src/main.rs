use vkjit_core::Ir;
use vkjit_rust::*;

fn main() {
    let x = [1., 2., 3.].as_slice();
    let x = F32::from(x);
    let y = x + x;

    let y = y.eval();

    println!("{:?}", y);
}
