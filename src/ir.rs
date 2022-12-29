use rspirv::binary::{Assemble, Disassemble};
use rspirv::spirv;
use screen_13::prelude::{vk, ComputePipeline, LazyPool, RenderGraph};
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

use crevice::std140::{self, AsStd140};

use crate::array::{self, Array};

#[derive(Debug, Clone, Copy)]
enum Const {
    Bool(bool),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
}

#[derive(Debug, Clone, Copy)]
enum Bop {
    Add,
}

#[derive(Debug, Clone, Copy)]
enum Op {
    Binding,
    Bop(Bop),
    Arange(usize),
    Const(Const),
    Zero,
    Access(usize),
    StructInit,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum VarType {
    Struct(Vec<VarType>),
    Void,
    Bool,
    UInt32,
    Int32,
    Float32,
}
impl VarType {
    pub fn name(&self) -> String {
        match self {
            VarType::Void => "Void".into(),
            VarType::Bool => "Bool".into(),
            VarType::UInt32 => "UInt32".into(),
            VarType::Int32 => "Int32".into(),
            VarType::Float32 => "Float32".into(),
            _ => unimplemented!(),
        }
    }
    pub fn stride(&self) -> usize {
        match self {
            VarType::Void => 0,
            VarType::Bool => bool::std140_size_static(),
            VarType::UInt32 => u32::std140_size_static(),
            VarType::Int32 => i32::std140_size_static(),
            VarType::Float32 => f32::std140_size_static(),
            _ => unimplemented!(),
        }
    }
    pub fn size(&self) -> usize {
        match self {
            VarType::Void => 0,
            VarType::Bool => bool::std140_size_static(),
            VarType::UInt32 => u32::std140_size_static(),
            VarType::Int32 => i32::std140_size_static(),
            VarType::Float32 => f32::std140_size_static(),
            _ => unimplemented!(),
        }
    }
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
            VarType::Struct(elems) => {
                let elems = elems
                    .iter()
                    .map(|elem| elem.to_spirv(b))
                    .collect::<Vec<_>>();
                b.type_struct(elems)
            }
            _ => unimplemented!(),
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
            _ => unimplemented!(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Var {
    op: Op,
    // Dependencies
    deps: Vec<usize>,
    pub array: Option<Arc<array::Array>>,
    ty: VarType,
}

#[derive(Debug)]
pub struct Ir {
    device: Arc<screen_13::driver::Device>,
    vars: Vec<Var>,
}
impl Ir {
    pub fn new(device: &Arc<screen_13::driver::Device>) -> Self {
        Self {
            device: device.clone(),
            vars: Vec::default(),
        }
    }
    pub fn array(&self, id: usize) -> &Arc<array::Array> {
        self.vars[id].array.as_ref().unwrap()
    }
    fn new_var(&mut self, op: Op, dep: Vec<usize>, ty: VarType) -> usize {
        self.push_var(Var {
            deps: dep,
            op,
            ty,
            array: None,
        })
    }
    fn push_var(&mut self, var: Var) -> usize {
        let id = self.vars.len();
        self.vars.push(var);
        id
    }
    pub fn var(&self, id: usize) -> &Var {
        &self.vars[id]
    }
    pub fn add(&mut self, lhs: usize, rhs: usize) -> usize {
        let lhs_ty = &self.vars[lhs].ty;
        let rhs_ty = &self.vars[rhs].ty;
        let ty = lhs_ty.max(rhs_ty);
        self.new_var(Op::Bop(Bop::Add), vec![lhs, rhs], ty.clone())
    }
    pub fn arange(&mut self, ty: VarType, num: usize) -> usize {
        self.new_var(Op::Arange(num), vec![], ty)
    }
    pub fn zero(&mut self, ty: VarType) -> usize {
        self.new_var(Op::Zero, vec![], ty)
    }
    pub fn struct_init(&mut self, vars: Vec<usize>) -> usize {
        let elems = vars
            .iter()
            .map(|id| {
                let var = &self.vars[*id];
                var.ty.clone()
            })
            .collect::<Vec<_>>();
        self.new_var(Op::StructInit, vars, VarType::Struct(elems))
    }
    pub fn const_f32(&mut self, val: f32) -> usize {
        self.new_var(Op::Const(Const::Float32(val)), vec![], VarType::Float32)
    }
    pub fn const_i32(&mut self, val: i32) -> usize {
        self.new_var(Op::Const(Const::Int32(val)), vec![], VarType::Int32)
    }
    pub fn const_u32(&mut self, val: u32) -> usize {
        self.new_var(Op::Const(Const::UInt32(val)), vec![], VarType::UInt32)
    }
    pub fn const_bool(&mut self, val: bool) -> usize {
        self.new_var(Op::Const(Const::Bool(val)), vec![], VarType::Bool)
    }
    pub fn array_f32(&mut self, data: &[f32]) -> usize {
        self.push_var(Var {
            ty: VarType::Float32,
            op: Op::Binding,
            deps: vec![],
            array: Some(Arc::new(Array::from_slice(
                &self.device,
                data,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            ))),
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Access {
    Read,
    Write,
}

#[derive(Debug, Clone, Copy)]
pub struct Binding {
    pub set: u32,
    pub binding: u32,
    pub access: Access,
}

pub struct Kernel {
    pub b: rspirv::dr::Builder,
    pub vars: HashMap<usize, u32>,
    pub num: Option<usize>,

    pub bindings: HashMap<usize, Binding>,
    pub arrays: HashMap<usize, u32>,
    pub array_structs: HashMap<VarType, u32>,
    pub structs: HashMap<Vec<VarType>, u32>,

    // Variables used by many kernels
    pub idx: Option<u32>,
    pub global_invocation_id: Option<u32>,
}
impl Debug for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kernel")
            .field("vars", &self.vars)
            .field("num", &self.num)
            .field("bindings", &self.bindings)
            .field("idx", &self.idx)
            .field("global_invocation_id", &self.global_invocation_id)
            .finish()
    }
}

impl Kernel {
    pub fn new() -> Self {
        Self {
            b: rspirv::dr::Builder::new(),
            vars: HashMap::default(),
            bindings: HashMap::default(),
            arrays: HashMap::default(),
            array_structs: HashMap::default(),
            structs: HashMap::default(),
            num: None,
            idx: None,
            global_invocation_id: None,
        }
    }
    fn binding(&mut self, id: usize, access: Access) -> Binding {
        if !self.bindings.contains_key(&id) {
            let binding = Binding {
                set: self.bindings.len() as u32,
                binding: 0,
                access,
            };
            self.bindings.insert(id, binding);
            binding
        } else {
            self.bindings[&id]
        }
    }
    fn record_array_struct_ty(&mut self, ty: &VarType) -> u32 {
        if self.array_structs.contains_key(ty) {
            return self.array_structs[ty];
        }
        let spv_ty = ty.to_spirv(&mut self.b);

        let ty_rta = self.b.type_runtime_array(spv_ty);
        let ty_struct = self.b.type_struct(vec![ty_rta]);
        let ty_struct_ptr = self
            .b
            .type_pointer(None, spirv::StorageClass::Uniform, ty_struct);

        let stride = ty.stride();
        self.b.decorate(
            ty_rta,
            spirv::Decoration::ArrayStride,
            vec![rspirv::dr::Operand::LiteralInt32(stride as u32)],
        );

        self.b
            .decorate(ty_struct, spirv::Decoration::BufferBlock, vec![]);
        self.b.member_decorate(
            ty_struct,
            0,
            spirv::Decoration::Offset,
            vec![rspirv::dr::Operand::LiteralInt32(0)],
        );
        self.b.name(ty_struct, ty.name());

        self.array_structs.insert(ty.clone(), ty_struct_ptr);
        ty_struct_ptr
    }
    fn record_binding(&mut self, id: usize, ir: &Ir, access: Access) -> u32 {
        if self.arrays.contains_key(&id) {
            return self.arrays[&id];
        }

        let var = &ir.vars[id];

        self.set_num(var.array.as_ref().unwrap().count());
        // https://shader-playground.timjones.io/3af32078f879d8599902e46b919dbfe3
        let binding = self.binding(id, access);
        let ty_struct_ptr = self.record_array_struct_ty(&var.ty);

        let st = self
            .b
            .variable(ty_struct_ptr, None, spirv::StorageClass::Uniform, None);
        self.b.decorate(
            st,
            spirv::Decoration::Binding,
            vec![rspirv::dr::Operand::LiteralInt32(binding.binding)],
        );
        self.b.decorate(
            st,
            spirv::Decoration::DescriptorSet,
            vec![rspirv::dr::Operand::LiteralInt32(binding.set)],
        );
        self.arrays.insert(id, st);
        st
    }
    ///
    /// Return a pointer to the binding at an index
    /// Note that idx is a spirv variable
    ///
    fn access_binding_at(&mut self, id: usize, ir: &Ir, idx: u32) -> u32 {
        println!("{}", id);
        let var = &ir.vars[id];
        let ty = var.ty.to_spirv(&mut self.b);
        let ty_int = self.b.type_int(32, 1);
        let int_0 = self.b.constant_u32(ty_int, 0);

        let ptr_ty = self.b.type_pointer(None, spirv::StorageClass::Uniform, ty);
        let ptr = self
            .b
            .access_chain(ptr_ty, None, self.arrays[&id], vec![int_0, idx])
            .unwrap();
        ptr
    }

    fn access_binding(&mut self, id: usize, ir: &Ir) -> u32 {
        let idx = self.idx.unwrap();
        self.access_binding_at(id, ir, idx)
    }
    fn set_num(&mut self, num: usize) {
        if let Some(num_) = self.num {
            assert!(
                num_ == num,
                "All variables in the kernel have to have the same number of elements!"
            )
        } else {
            self.num = Some(num);
        }
    }
    ///
    /// Traverse kernel and determine size. Panics if kernel size mismatches
    ///
    fn record_kernel_size(&mut self, id: usize, ir: &Ir) {
        let var = &ir.vars[id];
        match var.op {
            Op::Binding => {
                self.set_num(var.array.as_ref().unwrap().count());
            }
            Op::Arange(num) => {
                self.set_num(num);
            }
            _ => {
                for dep in var.deps.iter() {
                    self.record_kernel_size(*dep, ir);
                }
            }
        };
    }
    ///
    /// Records bindings before main function.
    ///
    fn record_bindings(&mut self, id: usize, ir: &Ir, access: Access) {
        if self.arrays.contains_key(&id) {
            return;
        }
        let var = &ir.vars[id];
        match var.op {
            Op::Binding => {
                self.record_binding(id, ir, access);
            }
            _ => {
                for dep in var.deps.iter() {
                    self.record_bindings(*dep, ir, access);
                }
            }
        };
    }
    ///
    /// Main record loop for recording variable operations.
    ///
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
            Op::Zero => {
                unimplemented!()
            }
            Op::Bop(bop) => {
                let lhs = self.record_var(var.deps[0], ir);
                let rhs = self.record_var(var.deps[0], ir);
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
            Op::Access(elem) => {
                //https://shader-playground.timjones.io/76ecd3898e50c0012918f6a080be6134
                let ty = var.ty.to_spirv(&mut self.b);
                let ptr_ty = self.b.type_pointer(None, spirv::StorageClass::Function, ty);
                let int_ty = self.b.type_int(32, 1);
                let idx = self.b.constant_u32(int_ty, elem as u32);
                let ptr = self
                    .b
                    .access_chain(ptr_ty, None, var.deps[0] as _, vec![idx])
                    .unwrap();
                self.b.load(ty, None, ptr, None, None).unwrap()
            }
            Op::StructInit => {
                let ty = var.ty.to_spirv(&mut self.b);
                let deps = var
                    .deps
                    .iter()
                    .map(|dep| self.record_var(*dep, ir))
                    .collect::<Vec<_>>();
                self.b.composite_construct(ty, None, deps).unwrap()
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
            Op::Binding => {
                let ty = var.ty.to_spirv(&mut self.b);
                let ptr = self.access_binding(id, ir);
                let ret = self.b.load(ty, None, ptr, None, None).unwrap();
                ret
            }
            _ => unimplemented!(),
        }
    }
    pub fn compile(&mut self, ir: &mut Ir, schedule: Vec<usize>) -> Vec<usize> {
        // Determine kernel size
        for id in schedule.iter() {
            self.record_kernel_size(*id, ir);
        }

        // Setup kernel with main function
        self.b.set_version(1, 3);
        self.b.capability(spirv::Capability::Shader);
        self.b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );

        // Setup default variables such as GlobalInvocationId
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

        // Reocrd bindings.
        // Bindings need to be prerecorded before main function.
        let schedule = schedule
            .iter()
            .map(|id1| {
                let ty = &ir.vars[*id1].ty.clone();
                let device = ir.device.clone();
                let id2 = ir.push_var(Var {
                    deps: vec![],
                    op: Op::Binding,
                    array: Some(Arc::new(Array::create(
                        &device,
                        ty,
                        self.num.expect("Could not determine size of kernel!"),
                        vk::BufferUsageFlags::STORAGE_BUFFER,
                    ))),
                    ty: ty.clone(),
                });
                self.record_bindings(*id1, ir, Access::Read);
                self.record_bindings(id2, ir, Access::Write);
                (*id1, id2)
            })
            .collect::<Vec<_>>();

        //

        println!("{:#?}", self.bindings);
        println!("{:#?}", self.arrays);

        // Setup main function
        let void = self.b.type_void();
        let voidf = self.b.type_function(void, vec![]);
        let main = self
            .b
            .begin_function(
                void,
                None,
                //rspirv::spirv::FunctionControl::DONT_INLINE | rspirv::spirv::FunctionControl::CONST,
                rspirv::spirv::FunctionControl::NONE,
                voidf,
            )
            .unwrap();
        self.b
            .execution_mode(main, spirv::ExecutionMode::LocalSize, vec![1, 1, 1]);
        self.b.begin_block(None).unwrap();

        // Load x component of GlobalInvocationId as index.
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

        // record scheduled variables

        let schedule = schedule
            .iter()
            .map(|(id1, id2)| {
                let spv_id = self.record_var(*id1, ir);
                (*id2, spv_id)
            })
            .collect::<Vec<_>>();

        // Write resulting variables
        let result = schedule
            .iter()
            .map(|(id2, spv_id)| {
                let ptr = self.access_binding(*id2, ir);
                self.b.store(ptr, *spv_id, None, None).unwrap();
                *id2
            })
            .collect::<Vec<_>>();

        // End main function
        self.b.ret().unwrap();
        self.b.end_function().unwrap();
        self.b.entry_point(
            rspirv::spirv::ExecutionModel::GLCompute,
            main,
            "main",
            vec![self.global_invocation_id.unwrap()],
        );

        result
    }
    pub fn record_render_graph(self, ir: &Ir, graph: &mut RenderGraph) {
        let module = self.b.module();
        println!("{}", module.disassemble());

        let spv = module.assemble();
        let pipeline = Arc::new(ComputePipeline::create(&ir.device, spv).unwrap());

        let nodes = self
            .bindings
            .iter()
            .map(|(id, _)| {
                let arr = ir.array(*id).clone();
                let node = graph.bind_node(&arr.buf);
                (*id, node)
            })
            .collect::<HashMap<_, _>>();

        let mut pass = graph.begin_pass("Eval kernel").bind_pipeline(&pipeline);
        for (id, binding) in self.bindings.iter() {
            println!("id={id}");
            match binding.access {
                Access::Read => {
                    println!("Read");
                    pass = pass.read_descriptor((binding.set, binding.binding), nodes[&id]);
                }
                Access::Write => {
                    println!("Write");
                    pass = pass.write_descriptor((binding.set, binding.binding), nodes[&id]);
                }
            }
        }
        let num = self.num.unwrap();
        pass.record_compute(move |compute, _| {
            compute.dispatch(num as u32, 1, 1);
        })
        .submit_pass();
    }
    pub fn execute(self, ir: &Ir, device: &Arc<screen_13::prelude::Device>) {
        let mut graph = RenderGraph::new();
        let mut pool = LazyPool::new(device);

        self.record_render_graph(&ir, &mut graph);

        graph.resolve().submit(&mut pool, 0).unwrap();

        unsafe { device.device_wait_idle().unwrap() };
    }
}
