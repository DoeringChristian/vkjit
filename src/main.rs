use vkjit_core::Ir;
use vkjit_rust::*;

fn main() {
    pretty_env_logger::init();

    let x = F32::from([1., 2., 3., 4., 5.]);
    let y = F32::from([0.; 6]);
    x.scatter_with(y, arange::<U32>(5), x.leq(3.));

    println!("{:#?}", IR.lock().unwrap());

    eval!(x);

    println!("{:?}", y);
}
