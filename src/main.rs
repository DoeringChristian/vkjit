use vkjit_core::Ir;
use vkjit_rust::*;

fn main() {
    pretty_env_logger::init();

    let x = Var::from(vec![1, 2, 3]);
    let i = arange(vkjit_core::VarType::U32, 3);
    // dbg!(i.id());
    let const2 = Var::from(2);
    let cond = i.clone().lt(const2);
    dbg!(cond.id());

    let x = gather_with(x, i.clone(), cond);

    println!("{:#?}", IR.lock().unwrap());

    eval!(x);

    println!("{:?}", x);
    println!("{:#?}", IR.lock().unwrap());
}
