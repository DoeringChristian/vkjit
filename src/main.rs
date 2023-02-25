use vkjit_core::Ir;
use vkjit_rust::*;

fn main() {
    pretty_env_logger::init();

    let y = {
        let x = Var::from(vec![1, 2, 3]);
        let x = x + 1;
        let alt = Var::from(vec![4, 4, 4]);
        // let c = Var::from(1);
        let y = select(x.lt(4), alt, 1);
        y
    };

    println!("{:#?}", IR.lock().unwrap());

    eval!(y);

    println!("{:?}", y);
    println!("{:#?}", IR.lock().unwrap());
}
