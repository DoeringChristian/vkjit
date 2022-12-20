use rspirv::spirv;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, Copy)]
enum Bop {
    Add,
}

#[derive(Debug, Clone, Copy)]
enum Const {
    Bool(bool),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
}

#[derive(Debug, Clone, Copy)]
enum Op {
    Buffer,
    Bop(Bop, usize, usize),
    Arange(u32),
    Const(Const),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum VarType {
    Void,
    Bool,
    UInt32,
    Int32,
    Float32,
}
impl VarType {
    fn to_spirv(&self, b: &mut rspirv::dr::Builder) -> u32 {
        match self {
            VarType::Void => b.type_void(),
            VarType::Bool => b.type_bool(),
            VarType::UInt32 => b.type_int(32, 0),
            VarType::Int32 => b.type_int(32, 1),
            VarType::Float32 => b.type_float(32),
        }
    }
}
impl From<VarType> for rspirv::sr::Type {
    fn from(ty: VarType) -> Self {
        match ty {
            VarType::Void => Self::Void,
            VarType::Bool => Self::Bool,
            VarType::UInt32 => Self::Int {
                width: 32,
                signedness: 0,
            },
            VarType::Int32 => Self::Int {
                width: 32,
                signedness: 1,
            },
            VarType::Float32 => Self::Float { width: 32 },
        }
    }
}

#[derive(Debug, Clone)]
struct Var {
    op: Op,
    buffer: Option<Arc<screen_13::driver::buffer::Buffer>>,
    ty: VarType,
}

#[derive(Debug, Default)]
pub struct Ir {
    vars: Vec<Var>,
}
impl Ir {
    fn new_var(&mut self, op: Op, ty: VarType) -> usize {
        let id = self.vars.len();
        self.vars.push(Var {
            op,
            ty,
            buffer: None,
        });
        id
    }
    pub fn add(&mut self, lhs: usize, rhs: usize) -> usize {
        let lhs_ty = self.vars[lhs].ty;
        let rhs_ty = self.vars[rhs].ty;
        let ty = lhs_ty.max(rhs_ty);
        self.new_var(Op::Bop(Bop::Add, lhs, rhs), ty)
    }
    pub fn arange(&mut self, ty: VarType, num: u32) -> usize {
        self.new_var(Op::Arange(num), ty)
    }
    pub fn const_f32(&mut self, val: f32) -> usize {
        self.new_var(Op::Const(Const::Float32(val)), VarType::Float32)
    }
    pub fn const_i32(&mut self, val: i32) -> usize {
        self.new_var(Op::Const(Const::Int32(val)), VarType::Int32)
    }
    pub fn const_u32(&mut self, val: u32) -> usize {
        self.new_var(Op::Const(Const::UInt32(val)), VarType::UInt32)
    }
    pub fn const_bool(&mut self, val: bool) -> usize {
        self.new_var(Op::Const(Const::Bool(val)), VarType::Bool)
    }
}
impl Ir {
    fn record_var(
        &self,
        id: usize,
        b: &mut rspirv::dr::Builder,
        vars: &mut HashMap<usize, u32>,
    ) -> u32 {
        if vars.contains_key(&id) {
            return vars[&id];
        }
        let var = &self.vars[id];
        match var.op {
            Op::Const(c) => {
                let ty = var.ty.to_spirv(b);
                let ret = match c {
                    Const::Bool(c) => {
                        if c {
                            b.constant_true(ty)
                        } else {
                            b.constant_false(ty)
                        }
                    }
                    Const::UInt32(c) => b.constant_u32(ty, c),
                    Const::Int32(c) => b.constant_u32(ty, unsafe { *(c as *const u32) }),
                    Const::Float32(c) => b.constant_f32(ty, c),
                    _ => unimplemented!(),
                };
                vars.insert(id, ret);
                ret
            }
            Op::Bop(bop, lhs, rhs) => {
                let lhs = self.record_var(lhs, b, vars);
                let rhs = self.record_var(rhs, b, vars);
                let ty = var.ty.to_spirv(b);
                let ret = match bop {
                    Bop::Add => match var.ty {
                        VarType::Int32 | VarType::UInt32 => b.i_add(ty, None, lhs, rhs).unwrap(),
                        VarType::Float32 => b.f_add(ty, None, lhs, rhs).unwrap(),
                        _ => panic!("Addition not defined for type {:?}", var.ty),
                    },
                };
                vars.insert(id, ret);
                ret
            }
            Op::Arange(num) => {
                let uint = b.type_int(32, 0);
                let v3uint = b.type_vector(uint, 3);
                let ptr_input_v3uint =
                    b.type_pointer(None, rspirv::spirv::StorageClass::Input, v3uint);
                let global_invocation_id = b.variable(
                    ptr_input_v3uint,
                    None,
                    rspirv::spirv::StorageClass::Input,
                    None,
                );
                b.decorate(
                    global_invocation_id,
                    spirv::Decoration::BuiltIn,
                    vec![rspirv::dr::Operand::BuiltIn(
                        spirv::BuiltIn::GlobalInvocationId,
                    )],
                );
                // Load x component of GlobalInvocationId
                let uint_0 = b.constant_u32(uint, 0);
                let uint = b.type_int(32, 0);
                let ptr_input_uint = b.type_pointer(None, spirv::StorageClass::Input, uint);
                let ptr = b
                    .access_chain(ptr_input_uint, None, global_invocation_id, vec![uint_0])
                    .unwrap();
                let idx = b.load(uint, None, ptr, None, None).unwrap();
                let ret = match var.ty {
                    VarType::UInt32 => idx,
                    VarType::Int32 => {
                        let ty = var.ty.to_spirv(b);
                        b.bitcast(ty, None, idx).unwrap()
                    }
                    VarType::Float32 => {
                        let ty = var.ty.to_spirv(b);
                        b.convert_u_to_f(ty, None, idx).unwrap()
                    }
                    _ => unimplemented!(),
                };
                vars.insert(id, ret);
                ret
            }
            _ => unimplemented!(),
        }
    }
    pub fn compile(&self, schedule: Vec<usize>) -> rspirv::dr::Module {
        let mut vars: HashMap<usize, u32> = HashMap::default();

        let mut b = rspirv::dr::Builder::new();

        // Setup kernel with main function
        b.set_version(1, 3);
        b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );

        let void = b.type_void();
        let voidf = b.type_function(void, vec![void]);
        let main = b
            .begin_function(
                void,
                None,
                rspirv::spirv::FunctionControl::DONT_INLINE | rspirv::spirv::FunctionControl::CONST,
                voidf,
            )
            .unwrap();
        b.begin_block(None).unwrap();

        for var in schedule {
            self.record_var(var, &mut b, &mut vars);
        }

        // End main function
        b.ret().unwrap();
        b.end_function().unwrap();
        b.entry_point(
            rspirv::spirv::ExecutionModel::GLCompute,
            main,
            "main",
            vec![],
        );
        b.module()
    }
}
