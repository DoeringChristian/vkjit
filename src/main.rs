use vkjit_core::Ir;
use vkjit_rust::*;

fn main() {
    let x = F32::from([1., 2., 3., 4., 5.]);
    let y = F32::from([0.; 5]);
    let x = x.select(0., x.leq(3.));
    let cond = x.leq(3.);
    x.scatter_with(y, arange::<U32>(5), cond);

    let x = x.eval();
    println!("test");

    println!("{:?}", y);
}
