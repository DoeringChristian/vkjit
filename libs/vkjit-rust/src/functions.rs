use vkjit_core::{AsVarType, VarId, VarType};

use crate::*;

pub fn zeros(ty: VarType) -> Var {
    Var::from(IR.lock().unwrap().zeros(ty))
}

pub fn arange(ty: VarType, num: usize) -> Var {
    Var::from(IR.lock().unwrap().arange(ty, num))
}

pub fn linspace(start: impl Into<Var>, stop: impl Into<Var>, num: usize) -> Var {
    let start = start.into();
    let stop = stop.into();
    let ret = Var::from(
        IR.lock()
            .unwrap()
            .linspace(start.ty(), start.id(), stop.id(), num),
    );
    ret
}
pub fn select(condition: impl Into<Var>, x: impl Into<Var>, y: impl Into<Var>) -> Var {
    let condition = condition.into();
    let x = x.into();
    let y = y.into();

    let ret = Var::from(IR.lock().unwrap().select(condition.id(), x.id(), y.id()));
    ret
}
pub fn gather(from: Var, idx: impl Into<Var>) -> Var {
    let idx = idx.into();
    let ret = Var::from(IR.lock().unwrap().gather(from.id(), idx.id(), None));
    drop(idx);
    drop(from);
    ret
}
pub fn gather_with(from: Var, idx: impl Into<Var>, condition: impl Into<Option<Var>>) -> Var {
    let idx = idx.into();
    let condition = condition.into();
    let cond = condition.as_ref().map(|condition| condition.id());

    log::trace!(
        "Gather from {:?} at {:?} with {:?}",
        from.id(),
        idx.id(),
        condition,
    );

    let mut ir = IR.lock().unwrap();
    let ret = Var::from(ir.gather(from.id(), idx.id(), cond));
    ret
}

#[macro_export]
macro_rules! eval {
    ($($var:expr)*) => {
        let schedule = [$($var.id()),*];
        $crate::eval_internal(&schedule);
    };
}

pub fn eval_internal(schedule: &[VarId]) {
    let mut ir = IR.lock().unwrap();
    ir.eval(&schedule);
}

#[cfg(test)]
mod test {
    use crate::*;
    #[test]
    fn test_scatter() {
        pretty_env_logger::init();

        let x = Var::from(vec![1., 2., 3.]);
        let mut y = Var::from(7.);
        let idx = arange(VarType::U32, 3);

        y.scatter(x.clone(), idx);

        eval!(y);

        assert_eq!(x.to_vec::<f32>(), vec![7., 7., 7.]);
    }
    #[test]
    fn setattr() {
        pretty_env_logger::init();

        let x = Var::from(vec![1., 2., 3.]);

        let mut st = zeros(VarType::Struct(vec![VarType::F32, VarType::F32]));

        st.setattr(x.clone(), 0);

        eval!(x);

        todo!()
    }
}
