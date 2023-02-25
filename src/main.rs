use vkjit_core::Ir;
use vkjit_rust::*;

fn main() {
    pretty_env_logger::init();

    let x = {
        let x = Var::from(vec![1, 2, 3]);
        let y = Var::from(1);

        let s = Var::from(&[x, y][..]);

        s.getattr(1)
    };

    println!("{:#?}", IR.lock().unwrap());

    eval!(x);

    println!("{:?}", x);
    println!("{:#?}", IR.lock().unwrap());
}
