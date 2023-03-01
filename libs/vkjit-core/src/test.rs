#[cfg(test)]
mod test {
    use std::sync::Arc;

    use crate::internal::Ir;
    use crate::vartype::VarType;
    use screen_13::prelude::*;

    #[test]
    fn test_linspace_f32() {
        let mut ir = Ir::new();

        let start = ir.const_f32(2.);
        let stop = ir.const_f32(4.);
        let x = ir.linspace(VarType::F32, start, stop, 4);

        ir.eval(&[x]);

        assert_eq!(ir.as_slice::<f32>(x), &[2., 2.5, 3., 3.5]);
    }
    #[test]
    fn test_linspace_eval2() {
        pretty_env_logger::init();
        let mut ir = Ir::new();

        let start = ir.const_f32(2.);
        let stop = ir.const_f32(4.);
        let x = ir.linspace(VarType::F32, start, stop, 4);

        ir.eval(&[x]);

        assert_eq!(ir.as_slice::<f32>(x), &[2., 2.5, 3., 3.5]);

        let start = ir.const_f32(10.);
        let stop = ir.const_f32(20.);

        let x = ir.linspace(VarType::F32, start, stop, 10);

        ir.eval(&[x]);
        assert_eq!(
            ir.as_slice::<f32>(x),
            &[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
        );
    }

    #[test]
    fn test_add_f32() {
        let mut ir = Ir::new();

        let x = ir.arange(VarType::F32, 3);
        let y = ir.arange(VarType::F32, 3);

        let z = ir.add(x, y);

        ir.eval(&[z]);

        assert_eq!(ir.as_slice::<f32>(z), &[0., 2., 4.]);
    }
    #[test]
    fn test_add_u32() {
        let mut ir = Ir::new();

        let x = ir.arange(VarType::U32, 3);
        let y = ir.arange(VarType::U32, 3);

        let z = ir.add(x, y);

        ir.eval(&[z]);

        assert_eq!(ir.as_slice::<u32>(z), &[0, 2, 4]);
    }
    #[test]
    fn test_add_i32() {
        let mut ir = Ir::new();

        let x = ir.arange(VarType::I32, 3);
        let y = ir.arange(VarType::I32, 3);

        let z = ir.add(x, y);

        ir.eval(&[z]);

        assert_eq!(ir.as_slice::<i32>(z), &[0, 2, 4]);
    }

    #[test]
    fn test_sub_f32() {
        let mut ir = Ir::new();

        let x = ir.array_f32(&[1., 2., 3.]);
        let y = ir.array_f32(&[0., 1., 2.]);

        let z = ir.sub(x, y);

        ir.eval(&[z]);

        assert_eq!(ir.as_slice::<f32>(z), &[1., 1., 1.]);
    }
    #[test]
    fn test_sub_u32() {
        let mut ir = Ir::new();

        let x = ir.array_u32(&[1, 2, 3]);
        let y = ir.array_u32(&[0, 1, 2]);

        let z = ir.sub(x, y);

        ir.eval(&[z]);

        assert_eq!(ir.as_slice::<u32>(z), &[1, 1, 1]);
    }
    #[test]
    fn test_sub_i32() {
        let mut ir = Ir::new();

        let x = ir.array_i32(&[0, 1, 2]);
        let y = ir.array_i32(&[1, 2, 3]);

        let z = ir.sub(x, y);

        ir.eval(&[z]);

        assert_eq!(ir.as_slice::<i32>(z), &[-1, -1, -1]);
    }

    #[test]
    fn test_scatter_f32() {
        let mut ir = Ir::new();

        let idx = ir.arange(VarType::U32, 3);
        let x = ir.array_f32(&[0., 1., 2.]);
        let c = ir.const_f32(1.);
        let x = ir.add(x, c);

        let y = ir.array_f32(&[0., 0., 0.]);

        let x = ir.scatter(x, y, idx, None);

        ir.eval(&[x]);

        assert_eq!(ir.as_slice::<f32>(y), &[1., 2., 3.]);
    }
    #[test]
    fn test_scatter_conditional() {
        let mut ir = Ir::new();

        let idx = ir.arange(VarType::U32, 3);
        let x = ir.array_f32(&[0., 1., 2., 3., 4.]);

        let y = ir.array_f32(&[0., 0., 0., 0., 0.]);

        let const3 = ir.const_u32(3);
        let cond = ir.lt(idx, const3);

        let x = ir.scatter(x, y, idx, Some(cond));

        ir.eval(&[x]);

        assert_eq!(ir.as_slice::<f32>(y), &[0., 1., 2., 0., 0.]);
    }

    #[test]
    fn cast_u32_to_f32() {
        let mut ir = Ir::new();

        let x = ir.arange(VarType::U32, 3);
        let y = ir.cast(x, &VarType::F32);

        ir.eval(&[y]);

        assert_eq!(ir.as_slice::<f32>(y), &[0., 1., 2.]);
    }

    #[test]
    fn autocast() {
        let mut ir = Ir::new();

        let x = ir.array_u32(&[1, 2]);
        let y = ir.const_i32(-1);

        let z = ir.add(x, y);

        ir.eval(&[z]);

        assert_eq!(ir.as_slice::<i32>(z), &[0, 1]);
    }

    #[test]
    fn dec_ref_count() {
        pretty_env_logger::init();
        let mut ir = Ir::new();

        let x = ir.array_f32(&[1., 2., 3.]);
        let c = ir.const_f32(1.);

        let y = ir.add(x, c);

        ir.dec_ref_count(c);
        ir.dec_ref_count(x);

        ir.eval(&[y]);

        assert_eq!(ir.vars.len(), 3);
        assert_eq!(ir.vars[0].ref_count, 0);
        assert_eq!(ir.arrays.len(), 1);
    }
}
