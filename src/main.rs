use vkjit_core::Ir;
use vkjit_rust::*;

fn main() {
    pretty_env_logger::init();

    let x = Var::from(vec![1., 2., 3.]);

    let mut st = zeros(VarType::Struct(vec![VarType::F32, VarType::F32]));

    st.setattr(x.clone(), 0);

    let y = st.getattr(0);

    eval!(y);

    println!("{:?}", y);
}
