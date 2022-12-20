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
    Array,
    Bop(Bop, usize, usize),
    Arange(usize),
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
    #[allow(unused)]
    pub fn from_rs<T: 'static>() -> Self {
        let ty_f32 = std::any::TypeId::of::<f32>();
        let ty_u32 = std::any::TypeId::of::<u32>();
        let ty_i32 = std::any::TypeId::of::<i32>();
        let ty_bool = std::any::TypeId::of::<bool>();
        match std::any::TypeId::of::<T>() {
            ty_f32 => Self::Float32,
            ty_u32 => Self::UInt32,
            ty_i32 => Self::Int32,
            ty_bool => Self::Bool,
            _ => unimplemented!(),
        }
    }
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
    pub fn arange(&mut self, ty: VarType, num: usize) -> usize {
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

pub struct Compiler {
    pub b: rspirv::dr::Builder,
    pub vars: HashMap<usize, u32>,
    pub num: Option<usize>,

    // Variables used by many kernels
    pub idx: Option<u32>,
    pub global_invocation_id: Option<u32>,
}

impl Compiler {
    pub fn new() -> Self {
        Self {
            b: rspirv::dr::Builder::new(),
            vars: HashMap::default(),
            num: None,
            idx: None,
            global_invocation_id: None,
        }
    }
    fn set_num(&mut self, num: usize) {
        if let Some(num_) = self.num {
            assert!(num_ == num)
        } else {
            self.num = Some(num);
        }
    }
    fn record_var(&mut self, id: usize, ir: &Ir) -> u32 {
        if self.vars.contains_key(&id) {
            return self.vars[&id];
        }
        let var = &ir.vars[id];
        match var.op {
            Op::Const(c) => {
                let ty = var.ty.to_spirv(&mut self.b);
                let ret = match c {
                    Const::Bool(c) => {
                        if c {
                            self.b.constant_true(ty)
                        } else {
                            self.b.constant_false(ty)
                        }
                    }
                    Const::UInt32(c) => self.b.constant_u32(ty, c),
                    Const::Int32(c) => self.b.constant_u32(ty, unsafe { *(c as *const u32) }),
                    Const::Float32(c) => self.b.constant_f32(ty, c),
                    _ => unimplemented!(),
                };
                self.vars.insert(id, ret);
                ret
            }
            Op::Bop(bop, lhs, rhs) => {
                let lhs = self.record_var(lhs, ir);
                let rhs = self.record_var(rhs, ir);
                let ty = var.ty.to_spirv(&mut self.b);
                let ret = match bop {
                    Bop::Add => match var.ty {
                        VarType::Int32 | VarType::UInt32 => {
                            self.b.i_add(ty, None, lhs, rhs).unwrap()
                        }
                        VarType::Float32 => self.b.f_add(ty, None, lhs, rhs).unwrap(),
                        _ => panic!("Addition not defined for type {:?}", var.ty),
                    },
                };
                self.vars.insert(id, ret);
                ret
            }
            Op::Arange(num) => {
                self.set_num(num);
                let ret = match var.ty {
                    VarType::UInt32 => self.idx.unwrap(),
                    VarType::Int32 => {
                        let ty = var.ty.to_spirv(&mut self.b);
                        self.b.bitcast(ty, None, self.idx.unwrap()).unwrap()
                    }
                    VarType::Float32 => {
                        let ty = var.ty.to_spirv(&mut self.b);
                        self.b.convert_u_to_f(ty, None, self.idx.unwrap()).unwrap()
                    }
                    _ => unimplemented!(),
                };
                self.vars.insert(id, ret);
                ret
            }
            Op::Array => {
                todo!()
            }
            _ => unimplemented!(),
        }
    }
    pub fn record_idx(&mut self, ir: &Ir) {
        let uint = self.b.type_int(32, 0);
        let v3uint = self.b.type_vector(uint, 3);
        let ptr_input_v3uint =
            self.b
                .type_pointer(None, rspirv::spirv::StorageClass::Input, v3uint);
        let global_invocation_id = self.b.variable(
            ptr_input_v3uint,
            None,
            rspirv::spirv::StorageClass::Input,
            None,
        );
        self.b.decorate(
            global_invocation_id,
            spirv::Decoration::BuiltIn,
            vec![rspirv::dr::Operand::BuiltIn(
                spirv::BuiltIn::GlobalInvocationId,
            )],
        );
        // Load x component of GlobalInvocationId
        let uint_0 = self.b.constant_u32(uint, 0);
        let uint = self.b.type_int(32, 0);
        let ptr_input_uint = self.b.type_pointer(None, spirv::StorageClass::Input, uint);
        let ptr = self
            .b
            .access_chain(ptr_input_uint, None, global_invocation_id, vec![uint_0])
            .unwrap();
        let idx = self.b.load(uint, None, ptr, None, None).unwrap();
        self.idx = Some(idx);
        self.global_invocation_id = Some(global_invocation_id);
    }
    pub fn compile(&mut self, ir: &Ir, schedule: Vec<usize>) {
        // Setup kernel with main function
        self.b.set_version(1, 3);
        self.b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );

        let void = self.b.type_void();
        let voidf = self.b.type_function(void, vec![void]);
        let main = self
            .b
            .begin_function(
                void,
                None,
                rspirv::spirv::FunctionControl::DONT_INLINE | rspirv::spirv::FunctionControl::CONST,
                voidf,
            )
            .unwrap();
        self.b.begin_block(None).unwrap();

        self.record_idx(ir);

        for var in schedule {
            self.record_var(var, ir);
        }

        // End main function
        self.b.ret().unwrap();
        self.b.end_function().unwrap();
        self.b.entry_point(
            rspirv::spirv::ExecutionModel::GLCompute,
            main,
            "main",
            vec![],
        );
    }
}
