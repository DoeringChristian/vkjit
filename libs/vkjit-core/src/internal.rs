use bytemuck::cast_slice;
use paste::paste;
use rspirv::binary::{Assemble, Disassemble};
use rspirv::spirv;
use screen_13::prelude::{vk, ComputePipeline, LazyPool, RenderGraph};
use std::any::TypeId;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::ops::Deref;
use std::sync::Arc;

use log::{error, trace, warn};

use crevice::std140::{self, AsStd140};

use crate::array::{self, Array};

use crate::iterators::{DepIterator, SeIterator};
use crate::vartype::*;

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
    Sub,
    Mul,
    Div,
    Lt,
    Gt,
    Eq,
    Leq,
    Geq,
    Neq,
}

impl Bop {
    fn eval_ty<'a>(self, lhs: &'a VarType, rhs: &'a VarType) -> VarType {
        match self {
            Self::Add | Self::Sub | Self::Mul | Self::Div => lhs.max(rhs).clone(),
            Self::Lt | Self::Gt | Self::Eq | Self::Leq | Self::Geq | Self::Neq => VarType::Bool,
            _ => unimplemented!(),
        }
    }
    fn eval_src_ty(self, lhs: &VarType, rhs: &VarType) -> VarType {
        match self {
            Self::Add => lhs.max(rhs).clone(),
            Self::Sub => lhs.max(rhs).clone(),
            Self::Div => VarType::F32,
            Self::Mul => VarType::F32,
            _ => unimplemented!(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Op {
    Binding,
    Bop(Bop),
    Arange(usize),
    Const(Const),
    GetAttr(usize),
    SetAttr(usize),
    StructInit, // Structs are sotred as pointers and StructInit returns a pointer to a struct
    Gather,
    Scatter,
    Select,
    Cast,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct VarId(usize);
impl Debug for VarId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl From<usize> for VarId {
    fn from(val: usize) -> Self {
        Self(val)
    }
}
impl VarId {
    pub fn get_id(&self) -> usize {
        self.0
    }
}

impl Deref for VarId {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub struct Var {
    op: Op,
    // Dependencies
    pub(crate) deps: Vec<VarId>,
    pub(crate) side_effects: Vec<VarId>,
    //pub array: Option<Arc<array::Array>>,
    ty: VarType,
}
impl Var {
    pub fn ty(&self) -> &VarType {
        &self.ty
    }
}

#[derive(Debug)]
pub struct Backend {
    device: Arc<screen_13::prelude::Device>,
    arrays: HashMap<VarId, array::Array>,
}

pub struct Ir {
    backend: Backend,
    vars: Vec<Var>,
    schedule: Vec<VarId>,
}

impl Debug for Ir {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut st = f.debug_struct("Ir");
        // st.field("backend", &self.backend)
        //     .field("vars", &self.vars)
        //     .field("schedule", &self.schedule);
        //
        // st.finish();

        for (i, x) in self.vars.iter().enumerate() {
            st.field(&format!("[{i}]"), x);
        }

        st.finish()
    }
}

macro_rules! bop {
    ($bop:ident) => {
        paste! {
        pub fn [<$bop:lower>](&mut self, lhs: VarId, rhs: VarId) -> VarId {
            let lhs_ty = &self.var( lhs ).ty;
            let rhs_ty = &self.var( rhs ).ty;

            // We perform the operation (i.e. lt or add) in the "largest" type
            let opty = lhs_ty.max(rhs_ty).clone();

            // The resulting type is not neccesarily the same (i.e. lt(f32, f32) -> bool)
            let ty = Bop::$bop.eval_ty(lhs_ty, rhs_ty);

            let lhs = self.cast(lhs, &opty);
            let rhs = self.cast(rhs, &opty);

            self.new_var(Op::Bop(Bop::$bop), vec![lhs, rhs], ty)
        }
        }
    };
}
impl Ir {
    pub fn new_sc13(device: &Arc<screen_13::driver::Device>) -> Self {
        Self {
            backend: Backend {
                device: device.clone(),
                arrays: HashMap::default(),
            },
            vars: Vec::default(),
            schedule: Vec::default(),
        }
    }
    pub fn new() -> Self {
        let cfg = screen_13::prelude::DriverConfig::new().build();
        let device = Arc::new(screen_13::prelude::Device::new(cfg).unwrap());
        // let sc13 = screen_13::prelude::EventLoop::new()
        //     .debug(true)
        //     .build()
        //     .unwrap();
        // let device = sc13.device.clone();
        Self {
            backend: Backend {
                device,
                arrays: HashMap::default(),
            },
            vars: Vec::default(),
            schedule: Vec::default(),
        }
    }
    pub fn array(&self, id: VarId) -> &array::Array {
        &self.backend.arrays[&id]
    }
    fn new_var(&mut self, op: Op, dependencies: Vec<VarId>, ty: VarType) -> VarId {
        self.push_var(Var {
            deps: dependencies,
            side_effects: Vec::new(),
            op,
            ty,
        })
    }
    fn push_var(&mut self, var: Var) -> VarId {
        let id = self.vars.len();
        self.vars.push(var);
        VarId(id)
    }
    pub fn var(&self, id: VarId) -> &Var {
        &self.vars[id.0]
    }
    pub fn var_mut(&mut self, id: VarId) -> &mut Var {
        &mut self.vars[id.0]
    }

    // Implement binary operations using bop macro
    bop!(Add);
    bop!(Sub);
    bop!(Mul);
    bop!(Div);
    bop!(Lt);
    bop!(Gt);
    bop!(Eq);
    bop!(Leq);
    bop!(Geq);
    bop!(Neq);

    pub fn select(&mut self, cond_id: VarId, lhs_id: VarId, rhs_id: VarId) -> VarId {
        let lhs_ty = &self.var(lhs_id).ty;
        let rhs_ty = &self.var(rhs_id).ty;
        assert!(lhs_ty == rhs_ty);
        self.new_var(Op::Select, vec![cond_id, lhs_id, rhs_id], lhs_ty.clone())
    }
    pub fn arange(&mut self, ty: VarType, num: usize) -> VarId {
        self.new_var(Op::Arange(num), vec![], ty)
    }
    pub fn linspace(&mut self, ty: VarType, start_id: VarId, stop_id: VarId, num: usize) -> VarId {
        let len = self.sub(stop_id, start_id);
        let x = self.arange(ty, num);
        let x = self.div(x, len);
        let x = self.add(x, start_id);
        x
    }
    pub fn zeros(&mut self, ty: VarType) -> VarId {
        match ty {
            VarType::Struct(elems) => {
                let elems = elems
                    .iter()
                    .map(|elem| self.zeros(elem.clone()))
                    .collect::<Vec<_>>();
                self.struct_init(elems)
            }
            VarType::Bool => self.const_bool(false),
            VarType::I32 => self.const_i32(0),
            VarType::U32 => self.const_u32(0),
            VarType::F32 => self.const_f32(0.),
            _ => unimplemented!(),
        }
    }
    pub fn ones(&mut self, ty: VarType) -> VarId {
        match ty {
            VarType::Struct(elems) => {
                let elems = elems
                    .iter()
                    .map(|elem| self.ones(elem.clone()))
                    .collect::<Vec<_>>();
                self.struct_init(elems)
            }
            VarType::Bool => self.const_bool(true),
            VarType::I32 => self.const_i32(1),
            VarType::U32 => self.const_u32(1),
            VarType::F32 => self.const_f32(1.),
            _ => unimplemented!(),
        }
    }
    pub fn cast(&mut self, src: VarId, ty: &VarType) -> VarId {
        let src_ty = &self.var(src).ty;
        if src_ty == ty {
            return src;
        } else {
            self.new_var(Op::Cast, vec![src], ty.clone())
        }
    }
    pub fn struct_init(&mut self, vars: Vec<VarId>) -> VarId {
        let elems = vars
            .iter()
            .map(|id| {
                let var = &self.var(*id);
                var.ty.clone()
            })
            .collect::<Vec<_>>();
        self.new_var(Op::StructInit, vars, VarType::Struct(elems))
    }
    pub fn const_f32(&mut self, val: f32) -> VarId {
        self.new_var(Op::Const(Const::Float32(val)), vec![], VarType::F32)
    }
    pub fn const_i32(&mut self, val: i32) -> VarId {
        self.new_var(Op::Const(Const::Int32(val)), vec![], VarType::I32)
    }
    pub fn const_u32(&mut self, val: u32) -> VarId {
        self.new_var(Op::Const(Const::UInt32(val)), vec![], VarType::U32)
    }
    pub fn const_bool(&mut self, val: bool) -> VarId {
        self.new_var(Op::Const(Const::Bool(val)), vec![], VarType::Bool)
    }
    pub fn array_f32(&mut self, data: &[f32]) -> VarId {
        let id = self.push_var(Var {
            ty: VarType::F32,
            op: Op::Binding,
            deps: vec![],
            side_effects: vec![],
        });
        self.backend.arrays.insert(
            id,
            Array::from_slice(
                &self.backend.device,
                data,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            ),
        );
        id
    }
    pub fn array_i32(&mut self, data: &[i32]) -> VarId {
        let id = self.push_var(Var {
            ty: VarType::I32,
            op: Op::Binding,
            deps: vec![],
            side_effects: vec![],
        });
        self.backend.arrays.insert(
            id,
            Array::from_slice(
                &self.backend.device,
                data,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            ),
        );
        id
    }
    pub fn array_u32(&mut self, data: &[u32]) -> VarId {
        let id = self.push_var(Var {
            ty: VarType::U32,
            op: Op::Binding,
            deps: vec![],
            side_effects: vec![],
        });
        self.backend.arrays.insert(
            id,
            Array::from_slice(
                &self.backend.device,
                data,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            ),
        );
        id
    }
    pub fn getattr(&mut self, src_id: VarId, idx: usize) -> VarId {
        let src = &self.var(src_id);
        let ty = match src.ty {
            VarType::Struct(ref elems) => elems[idx].clone(),
            _ => unimplemented!(),
        };
        self.new_var(Op::GetAttr(idx), vec![src_id], ty)
    }
    pub fn setattr(&mut self, dst_id: VarId, src_id: VarId, idx: usize) {
        let src = &self.var(src_id);
        let ty = src.ty.clone();
        let var = self.new_var(Op::SetAttr(idx), vec![src_id], src.ty.clone());
        let dst = self.var_mut(dst_id);
        dst.side_effects.push(var);
    }
    pub fn gather(&mut self, src_id: VarId, idx_id: VarId) -> VarId {
        let src = &self.var(src_id);
        self.new_var(Op::Gather, vec![src_id, idx_id], src.ty.clone())
    }
    pub fn scatter(
        &mut self,
        src_id: VarId,
        dst_id: VarId,
        idx_id: VarId,
        active_id: Option<VarId>,
    ) {
        let src = &self.var(src_id);
        let mut deps = vec![dst_id, idx_id];
        active_id.and_then(|id| {
            deps.push(id);
            Some(())
        });
        let var = self.new_var(Op::Scatter, deps, src.ty.clone());
        self.var_mut(src_id).side_effects.push(var);
    }
    pub fn is_buffer(&self, id: &VarId) -> bool {
        self.backend.arrays.contains_key(id)
    }
    pub fn str(&self, id: VarId) -> String {
        let var = &self.var(id);
        let slice = screen_13::prelude::Buffer::mapped_slice(&self.backend.arrays[&id].buf);
        match var.ty {
            VarType::F32 => {
                format!("{:?}", cast_slice::<_, f32>(slice))
            }
            VarType::U32 => {
                format!("{:?}", cast_slice::<_, u32>(slice))
            }
            VarType::I32 => {
                format!("{:?}", cast_slice::<_, i32>(slice))
            }
            VarType::Bool => {
                format!("{:?}", cast_slice::<_, u8>(slice))
            }
            _ => format!("Undefined Type!"),
        }
    }
    pub fn print_buffer(&self, id: VarId) {
        let var = &self.var(id);
        let slice = screen_13::prelude::Buffer::mapped_slice(&self.backend.arrays[&id].buf);
        match var.ty {
            VarType::F32 => {
                println!("{:?}", cast_slice::<_, f32>(slice))
            }
            VarType::U32 => {
                println!("{:?}", cast_slice::<_, u32>(slice))
            }
            VarType::I32 => {
                println!("{:?}", cast_slice::<_, i32>(slice))
            }
            VarType::Bool => {
                println!("{:?}", cast_slice::<_, u8>(slice))
            }
            _ => unimplemented!(),
        }
    }
    pub fn as_slice<T: bytemuck::Pod>(&self, id: VarId) -> &[T] {
        let var = &self.var(id);
        let slice = screen_13::prelude::Buffer::mapped_slice(&self.backend.arrays[&id].buf);
        assert!(var.ty.type_id() == TypeId::of::<T>());
        cast_slice(slice)
    }
    pub fn schedule(&mut self, schedule: &[VarId]) {
        self.schedule.extend_from_slice(schedule);
    }
    pub fn eval(&mut self, schedule: &[VarId]) {
        self.schedule.extend_from_slice(schedule);
        #[cfg(test)]
        trace!("Compiling Kernel...");
        trace!("Internal Representation: {:#?}", self);
        let mut k = Kernel::new();
        let dst = k.compile(self);

        // Record render graph
        trace!("Recording Render Graph...");
        let mut graph = RenderGraph::new();
        let mut pool = LazyPool::new(&self.backend.device);

        let module = k.b.module();
        // #[cfg(test)]
        trace!("{}", module.disassemble());

        let spv = module.assemble();
        let pipeline = Arc::new(
            ComputePipeline::create(
                &self.backend.device,
                screen_13::prelude::ComputePipelineInfo::default(),
                screen_13::prelude::Shader::new_compute(spv),
            )
            .unwrap(),
        );

        // Collect nodes and corresponding bindings
        trace!("Collecting Nodes and Bindings...");
        let mut nodes = k
            .bindings
            .iter()
            .map(|(id, binding)| {
                let arr = self.array(*id).clone();
                let node = graph.bind_node(&arr.buf);
                (*binding, node)
            })
            .collect::<Vec<_>>();
        nodes.extend(dst.iter().map(|(binding, arr)| {
            let node = graph.bind_node(&arr.buf);
            (*binding, node)
        }));

        let mut pass = graph.begin_pass("Eval kernel").bind_pipeline(&pipeline);
        for (binding, node) in nodes {
            match binding.access {
                Access::Read => {
                    trace!("Binding buffer to {:?}", binding);
                    pass = pass.read_descriptor((binding.set, binding.binding), node);
                }
                Access::Write => {
                    trace!("Binding buffer to {:?}", binding);
                    pass = pass.write_descriptor((binding.set, binding.binding), node);
                }
            }
        }
        trace!("Recording Compute Pass...");
        let num = k.num.unwrap();
        pass.record_compute(move |compute, _| {
            compute.dispatch(num as u32, 1, 1);
        })
        .submit_pass();

        trace!("Resolving Graph...");
        graph.resolve().submit(&mut pool, 0).unwrap();

        trace!("Executing Computations...");
        unsafe { self.backend.device.device_wait_idle().unwrap() };

        trace!("Overwriting Evaluated Variables...");
        // Overwrite variables
        for (i, (_, arr)) in dst.into_iter().enumerate() {
            let id = self.schedule[i];
            *self.var_mut(id) = Var {
                op: Op::Binding,
                deps: vec![],
                side_effects: vec![],
                ty: arr.ty.clone(),
            };
            self.backend.arrays.insert(id, arr);
        }
    }

    pub fn iter_dep(&self, root: &[VarId]) -> DepIterator {
        DepIterator {
            ir: self,
            stack: Vec::from(root),
            discovered: HashSet::default(),
        }
    }
    pub fn iter_se(&self, root: &[VarId]) -> SeIterator {
        SeIterator {
            ir: self,
            stack: Vec::from(root),
            discovered: HashSet::default(),
        }
    }
    // Composite operations
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
    pub op_results: HashMap<VarId, u32>,
    pub vars: HashMap<VarId, u32>,
    pub num: Option<usize>,

    pub bindings: HashMap<VarId, Binding>,
    pub arrays: HashMap<VarId, u32>,
    pub array_structs: HashMap<VarType, u32>,
    pub structs: HashMap<Vec<VarType>, u32>,

    // Variables used by many kernels
    pub idx: Option<u32>,
    pub global_invocation_id: Option<u32>,

    pub traversal_set: HashSet<VarId>,
}
impl Debug for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kernel")
            .field("vars", &self.op_results)
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
            op_results: HashMap::default(),
            vars: HashMap::default(),
            bindings: HashMap::default(),
            arrays: HashMap::default(),
            array_structs: HashMap::default(),
            structs: HashMap::default(),
            num: None,
            idx: None,
            global_invocation_id: None,
            traversal_set: HashSet::default(),
        }
    }
    fn binding(&mut self, id: VarId, access: Access) -> Binding {
        if !self.bindings.contains_key(&id) {
            let binding = Binding {
                set: 0,
                binding: self.bindings.len() as u32,
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
    fn record_binding(&mut self, id: VarId, ir: &Ir, access: Access) -> u32 {
        if self.arrays.contains_key(&id) {
            return self.arrays[&id];
        }

        let var = &ir.var(id);

        //self.set_num(var.array.as_ref().unwrap().count());
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
    fn access_binding_at(&mut self, id: VarId, ir: &Ir, idx: u32) -> u32 {
        let var = &ir.var(id);
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

    fn access_binding(&mut self, id: VarId, ir: &Ir) -> u32 {
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
    fn record_kernel_size(&mut self, id: VarId, ir: &Ir) {
        if self.num.is_some() {
            return;
        }
        let var = &ir.var(id);
        match var.op {
            Op::Binding => {
                self.set_num(ir.backend.arrays[&id].count());
            }
            Op::Arange(num) => {
                self.set_num(num);
            }
            _ => {
                for id in var.deps.iter() {
                    self.record_kernel_size(*id, ir);
                }
            }
        };
    }
    ///
    /// Records bindings before main function.
    ///
    fn record_bindings(&mut self, schedule: &[VarId], ir: &Ir, access: Access) {
        for id in ir.iter_se(schedule) {
            let var = &ir.var(id);
            match var.op {
                Op::Binding => {
                    self.record_binding(id, ir, access);
                }
                _ => {}
            }
        }
    }
    fn record_if<F>(&mut self, conditional_id: u32, mut f: F)
    where
        F: FnMut(&mut Self),
    {
        let true_label_id = self.b.id();
        let end_label_id = self.b.id();

        // According to spirv OpSelectionMerge should be second to last
        // instruction in block. Rspirv however ends block with
        // selection_merge. Therefore, we insert the instruction by hand.
        self.b
            .insert_into_block(
                rspirv::dr::InsertPoint::End,
                rspirv::dr::Instruction::new(
                    spirv::Op::SelectionMerge,
                    None,
                    None,
                    vec![
                        rspirv::dr::Operand::IdRef(end_label_id),
                        rspirv::dr::Operand::SelectionControl(spirv::SelectionControl::NONE),
                    ],
                ),
            )
            .unwrap();
        self.b
            .branch_conditional(conditional_id, true_label_id, end_label_id, None)
            .unwrap();
        self.b.begin_block(Some(true_label_id)).unwrap();

        f(self);

        self.b.branch(end_label_id).unwrap();
        self.b.begin_block(Some(end_label_id)).unwrap();
    }
    fn record_ifelse<F, E>(&mut self, conditional_id: u32, mut f: F, mut e: E)
    where
        F: FnMut(&mut Self),
        E: FnMut(&mut Self),
    {
        let true_label_id = self.b.id();
        let false_label_id = self.b.id();
        let end_label_id = self.b.id();

        // According to spirv OpSelectionMerge should be second to last
        // instruction in block. Rspirv however ends block with
        // selection_merge. Therefore, we insert the instruction by hand.
        self.b
            .insert_into_block(
                rspirv::dr::InsertPoint::End,
                rspirv::dr::Instruction::new(
                    spirv::Op::SelectionMerge,
                    None,
                    None,
                    vec![
                        rspirv::dr::Operand::IdRef(end_label_id),
                        rspirv::dr::Operand::SelectionControl(spirv::SelectionControl::NONE),
                    ],
                ),
            )
            .unwrap();
        self.b
            .branch_conditional(conditional_id, true_label_id, false_label_id, None)
            .unwrap();
        self.b.begin_block(Some(true_label_id)).unwrap();

        f(self);

        self.b.branch(end_label_id).unwrap();
        self.b.begin_block(Some(false_label_id)).unwrap();

        e(self);

        self.b.branch(end_label_id).unwrap();
        self.b.begin_block(Some(end_label_id)).unwrap();
    }
    ///
    /// Main record loop for recording variable operations.
    ///
    fn record_ops(&mut self, id: VarId, ir: &Ir) -> u32 {
        if self.op_results.contains_key(&id) {
            return self.op_results[&id];
        }
        let var = &ir.var(id);
        let ret = match var.op {
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
                    Const::Int32(c) => self
                        .b
                        .constant_u32(ty, unsafe { *(&c as *const i32 as *const u32) }),
                    Const::Float32(c) => self.b.constant_f32(ty, c),
                    _ => unimplemented!(),
                };
                self.op_results.insert(id, ret);
                ret
            }
            Op::Bop(bop) => {
                let lhs_ty = &ir.var(var.deps[0]).ty;
                let rhs_ty = &ir.var(var.deps[1]).ty;
                let lhs = self.record_ops(var.deps[0], ir);
                let rhs = self.record_ops(var.deps[1], ir);
                let ty = var.ty.to_spirv(&mut self.b);
                let ret = match bop {
                    Bop::Add => match var.ty {
                        VarType::I32 | VarType::U32 => self.b.i_add(ty, None, lhs, rhs).unwrap(),
                        VarType::F32 => self.b.f_add(ty, None, lhs, rhs).unwrap(),
                        _ => panic!("Addition not defined for type {:?}", var.ty),
                    },
                    Bop::Sub => match var.ty {
                        VarType::I32 | VarType::U32 => self.b.i_sub(ty, None, lhs, rhs).unwrap(),
                        VarType::F32 => self.b.f_sub(ty, None, lhs, rhs).unwrap(),
                        _ => panic!("Addition not defined for type {:?}", var.ty),
                    },
                    Bop::Mul => match var.ty {
                        VarType::I32 | VarType::U32 => self.b.i_mul(ty, None, lhs, rhs).unwrap(),
                        VarType::F32 => self.b.f_mul(ty, None, lhs, rhs).unwrap(),
                        _ => panic!("Addition not defined for type {:?}", var.ty),
                    },
                    Bop::Div => match var.ty {
                        VarType::I32 => self.b.s_div(ty, None, lhs, rhs).unwrap(),
                        VarType::U32 => self.b.u_div(ty, None, lhs, rhs).unwrap(),
                        VarType::F32 => self.b.f_div(ty, None, lhs, rhs).unwrap(),
                        _ => panic!("Addition not defined for type {:?}", var.ty),
                    },
                    Bop::Lt => match lhs_ty {
                        VarType::F32 => self.b.f_ord_less_than(ty, None, lhs, rhs).unwrap(),
                        VarType::I32 => self.b.s_less_than(ty, None, lhs, rhs).unwrap(),
                        VarType::U32 => self.b.u_less_than(ty, None, lhs, rhs).unwrap(),
                        _ => unimplemented!(),
                    },
                    Bop::Gt => match lhs_ty {
                        VarType::F32 => self.b.f_ord_greater_than(ty, None, lhs, rhs).unwrap(),
                        VarType::I32 => self.b.s_greater_than(ty, None, lhs, rhs).unwrap(),
                        VarType::U32 => self.b.u_greater_than(ty, None, lhs, rhs).unwrap(),
                        _ => unimplemented!(),
                    },
                    Bop::Eq => match lhs_ty {
                        VarType::F32 => self.b.f_ord_equal(ty, None, lhs, rhs).unwrap(),
                        VarType::I32 | VarType::U32 => self.b.i_equal(ty, None, lhs, rhs).unwrap(),
                        _ => unimplemented!(),
                    },
                    Bop::Leq => match lhs_ty {
                        VarType::F32 => self.b.f_ord_less_than_equal(ty, None, lhs, rhs).unwrap(),
                        VarType::I32 => self.b.s_less_than_equal(ty, None, lhs, rhs).unwrap(),
                        VarType::U32 => self.b.u_less_than_equal(ty, None, lhs, rhs).unwrap(),
                        _ => unimplemented!(),
                    },
                    Bop::Geq => match lhs_ty {
                        VarType::F32 => {
                            self.b.f_ord_greater_than_equal(ty, None, lhs, rhs).unwrap()
                        }
                        VarType::I32 => self.b.s_greater_than_equal(ty, None, lhs, rhs).unwrap(),
                        VarType::U32 => self.b.u_greater_than_equal(ty, None, lhs, rhs).unwrap(),
                        _ => unimplemented!(),
                    },
                    Bop::Neq => match lhs_ty {
                        VarType::F32 => self.b.f_ord_not_equal(ty, None, lhs, rhs).unwrap(),
                        VarType::I32 | VarType::U32 => {
                            self.b.i_not_equal(ty, None, lhs, rhs).unwrap()
                        }
                        _ => unimplemented!(),
                    },
                    _ => unimplemented!(),
                };
                self.op_results.insert(id, ret);
                ret
            }
            Op::Cast => {
                let src_ty = &ir.var(var.deps[0]).ty;
                let src_spv = self.record_ops(var.deps[0], ir);
                trace!("Casting {:?} -> {:?}", src_ty, var.ty);
                match src_ty {
                    VarType::F32 => match var.ty {
                        VarType::F32 => src_spv,
                        VarType::U32 => {
                            let ty = var.ty.to_spirv(&mut self.b);
                            self.b.convert_u_to_f(ty, None, src_spv).unwrap()
                        }
                        VarType::I32 => {
                            let ty = var.ty.to_spirv(&mut self.b);
                            self.b.convert_s_to_f(ty, None, src_spv).unwrap()
                        }
                        _ => unimplemented!(),
                    },
                    VarType::U32 => match var.ty {
                        VarType::U32 | VarType::I32 => src_spv,
                        VarType::F32 => {
                            let ty = var.ty.to_spirv(&mut self.b);
                            self.b.convert_u_to_f(ty, None, src_spv).unwrap()
                        }
                        _ => unimplemented!(),
                    },
                    VarType::I32 => match var.ty {
                        VarType::U32 | VarType::I32 => src_spv,
                        VarType::F32 => {
                            let ty = var.ty.to_spirv(&mut self.b);
                            self.b.convert_s_to_f(ty, None, src_spv).unwrap()
                        }
                        _ => unimplemented!(),
                    },
                    _ => unimplemented!(),
                }
            }
            Op::GetAttr(elem) => {
                //https://shader-playground.timjones.io/76ecd3898e50c0012918f6a080be6134
                let ty = var.ty.to_spirv(&mut self.b);
                let ptr_ty = self.b.type_pointer(None, spirv::StorageClass::Function, ty);
                let int_ty = self.b.type_int(32, 1);
                let idx = self.b.constant_u32(int_ty, elem as u32);

                let src = self.record_ops(var.deps[0], ir);

                let ptr = self.b.access_chain(ptr_ty, None, src, vec![idx]).unwrap();
                self.b.load(ty, None, ptr, None, None).unwrap()
            }
            Op::SetAttr(elem) => 0, // TODO: Better return
            Op::StructInit => {
                let ty = var.ty.to_spirv(&mut self.b);
                let ty_ptr = self.b.type_pointer(None, spirv::StorageClass::Function, ty);
                let deps = var
                    .deps
                    .iter()
                    .map(|dep| self.record_ops(*dep, ir))
                    .collect::<Vec<_>>();
                let object = self.b.composite_construct(ty, None, deps).unwrap();
                let ptr = self.vars[&id];
                self.b.store(ptr, object, None, None).unwrap();
                ptr
            }
            Op::Gather => match ir.var(var.deps[0]).op {
                Op::Binding => {
                    let ty = var.ty.to_spirv(&mut self.b);
                    let idx = self.record_ops(var.deps[1], ir);
                    let ptr = self.access_binding_at(var.deps[0], ir, idx);
                    self.b.load(ty, None, ptr, None, None).unwrap()
                }
                _ => panic!("Can only gather from buffer!"),
            },
            Op::Scatter => 0, // TODO: better return
            Op::Arange(num) => {
                //self.set_num(num);
                let ret = match var.ty {
                    VarType::U32 => self.idx.unwrap(),
                    VarType::I32 => {
                        let ty = var.ty.to_spirv(&mut self.b);
                        self.b.bitcast(ty, None, self.idx.unwrap()).unwrap()
                    }
                    VarType::F32 => {
                        let ty = var.ty.to_spirv(&mut self.b);
                        self.b.convert_u_to_f(ty, None, self.idx.unwrap()).unwrap()
                    }
                    _ => unimplemented!(),
                };
                self.op_results.insert(id, ret);
                ret
            }
            Op::Binding => {
                let ty = var.ty.to_spirv(&mut self.b);
                let ptr = self.access_binding(id, ir);
                let ret = self.b.load(ty, None, ptr, None, None).unwrap();
                ret
            }
            Op::Select => {
                let ty = var.ty.to_spirv(&mut self.b);
                let cond_id = self.record_ops(var.deps[0], ir);
                let lhs_id = self.record_ops(var.deps[1], ir);
                let rhs_id = self.record_ops(var.deps[2], ir);
                let ptr = self.vars[&id];
                self.record_ifelse(
                    cond_id,
                    |s| {
                        s.b.store(ptr, lhs_id, None, None).unwrap();
                    },
                    |s| {
                        s.b.store(ptr, rhs_id, None, None).unwrap();
                    },
                );
                self.b.load(ty, None, ptr, None, None).unwrap()
            }
            _ => unimplemented!(),
        };
        self.op_results.insert(id, ret);
        // Evaluate side effects like setattr
        for id in var.side_effects.iter() {
            let se_spv = self.record_ops(*id, ir);
            let se = &ir.var(*id);
            match se.op {
                Op::SetAttr(elem) => {
                    let ty = se.ty.to_spirv(&mut self.b);
                    let ptr_ty = self.b.type_pointer(None, spirv::StorageClass::Function, ty);
                    let int_ty = self.b.type_int(32, 1);
                    let idx = self.b.constant_u32(int_ty, elem as u32);

                    let src = self.record_ops(se.deps[0], ir);

                    let ptr = self.b.access_chain(ptr_ty, None, ret, vec![idx]).unwrap();
                    self.b.store(ptr, src, None, None).unwrap();
                }
                Op::Scatter => match ir.var(se.deps[0]).op {
                    Op::Binding => {
                        let ty = se.ty.to_spirv(&mut self.b);
                        let idx = self.record_ops(se.deps[1], ir);
                        let ptr = self.access_binding_at(se.deps[0], ir, idx);

                        if se.deps.len() >= 3 {
                            let condition_id = self.record_ops(se.deps[2], ir);
                            self.record_if(condition_id, |s| {
                                s.b.store(ptr, ret, None, None).unwrap();
                            });
                        } else {
                            self.b.store(ptr, ret, None, None).unwrap();
                        }
                    }
                    _ => panic!("Cannot scatter into non buffer variable!"),
                },
                _ => {}
            }
        }
        ret
    }
    ///
    /// Record variables needed to store structs and variables for select.
    ///
    pub fn record_spv_vars(&mut self, schedule: &[VarId], ir: &Ir) {
        for id in ir.iter_dep(schedule) {
            let var = &ir.var(id);
            let ty = var.ty.to_spirv(&mut self.b);
            let ty_ptr = self.b.type_pointer(None, spirv::StorageClass::Function, ty);

            match var.op {
                Op::StructInit | Op::Select => {
                    let var = self
                        .b
                        .variable(ty_ptr, None, spirv::StorageClass::Function, None);
                    self.vars.insert(id, var);
                }
                _ => {}
            };
        }
    }
    pub fn compile(&mut self, ir: &Ir) -> Vec<(Binding, Array)> {
        trace!("Compiling Kernel...");
        // Determine kernel size
        trace!("Determining Kernel size...");
        for id in ir.schedule.iter() {
            self.record_kernel_size(*id, ir);
        }

        trace!("Kernel Configuration");
        self.b.set_version(1, 3);
        self.b.capability(spirv::Capability::Shader);
        self.b.memory_model(
            rspirv::spirv::AddressingModel::Logical,
            rspirv::spirv::MemoryModel::Simple,
        );

        // Setup default variables such as GlobalInvocationId
        trace!("Setting Up Default Variables...");
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

        trace!("Creating Destination Buffers and Bindings...");
        let dst = ir
            .schedule
            .iter()
            .enumerate()
            .map(|(i, id)| {
                let ty = &ir.var(*id).ty.clone();

                // Record dst bindings

                let ty_struct_ptr = self.record_array_struct_ty(ty);

                let binding = Binding {
                    set: 1,
                    binding: i as _,
                    access: Access::Write,
                };
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

                // Create dst arrays
                let arr = Array::create(
                    &ir.backend.device,
                    ty,
                    self.num.expect("Could not determine size of kernel!"),
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                );
                (arr, binding, st)
            })
            .collect::<Vec<_>>();

        // Record bindings for dependencies and side-effects TODO: scatter needs write!
        trace!("Recording Bindings for Source Variables...");
        self.record_bindings(&ir.schedule, ir, Access::Read);

        // Setup main function
        trace!("Recording main function...");
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

        trace!("Recording Variables...");
        self.record_spv_vars(&ir.schedule, ir);

        // Load x component of GlobalInvocationId as index.
        trace!("Recording GlobalInvocationId...");
        let uint = self.b.type_int(32, 0);
        let uint_0 = self.b.constant_u32(uint, 0);
        let ptr_input_uint = self.b.type_pointer(None, spirv::StorageClass::Input, uint);
        let ptr = self
            .b
            .access_chain(ptr_input_uint, None, global_invocation_id, vec![uint_0])
            .unwrap();
        let idx = self.b.load(uint, None, ptr, None, None).unwrap();
        self.idx = Some(idx);
        self.global_invocation_id = Some(global_invocation_id);

        // record scheduled variables

        trace!("Recording Operations...");
        let schedule_spv = ir
            .schedule
            .iter()
            .map(|id| {
                let spv_id = self.record_ops(*id, ir);
                spv_id
            })
            .collect::<Vec<_>>();

        // Write resulting variables
        trace!("Recording Write to Destination...");
        let dst = dst
            .into_iter()
            .enumerate()
            .map(|(i, (arr, binding, st))| {
                let ty = arr.ty.to_spirv(&mut self.b);
                let ty_int = self.b.type_int(32, 1);
                let int_0 = self.b.constant_u32(ty_int, 0);

                let ptr_ty = self.b.type_pointer(None, spirv::StorageClass::Uniform, ty);
                let ptr = self
                    .b
                    .access_chain(ptr_ty, None, st, vec![int_0, self.idx.unwrap()])
                    .unwrap();
                self.b.store(ptr, schedule_spv[i], None, None).unwrap();
                (binding, arr)
            })
            .collect::<Vec<_>>();

        // End main function
        trace!("Recording End main function...");
        self.b.ret().unwrap();
        self.b.end_function().unwrap();
        self.b.entry_point(
            rspirv::spirv::ExecutionModel::GLCompute,
            main,
            "main",
            vec![self.global_invocation_id.unwrap()],
        );

        dst
    }
}
