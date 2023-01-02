use vkjit_core::Ir;
use vkjit_rust::*;

fn main() {
    let x = F32::from([1., 2., 3., 4., 5.]);
    let z = x.select(0., x.lt(3.));

    let z = z.eval();

    println!("{:?}", z);
}
