#[cfg(test)]
mod test {
    use std::sync::Arc;

    use crate::internal::{Ir, VarType};
    use screen_13::prelude::*;

    #[test]
    fn test_add() {
        let mut ir = Ir::new();

        let x = ir.arange(VarType::Float32, 3);
        let y = ir.arange(VarType::Float32, 3);

        let z = ir.add(x, y);

        let res = ir.eval(vec![z]);

        assert_eq!(ir.as_slice::<f32>(res[0]), &[0., 2., 4.]);
    }
}
