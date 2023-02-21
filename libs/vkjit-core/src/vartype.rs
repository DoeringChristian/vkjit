use crevice::std140::AsStd140;
use std::any::TypeId;

pub trait AsVarType {
    fn as_var_type() -> VarType;
}

impl AsVarType for u32 {
    fn as_var_type() -> VarType {
        VarType::U32
    }
}
impl AsVarType for i32 {
    fn as_var_type() -> VarType {
        VarType::I32
    }
}
impl AsVarType for f32 {
    fn as_var_type() -> VarType {
        VarType::F32
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum VarType {
    Struct(Vec<VarType>), // Structs are stored as pointer and StructInit returns a pointer to a
    // struct
    Void,
    Bool,
    U32,
    I32,
    F32,
}
impl VarType {
    pub fn name(&self) -> String {
        match self {
            VarType::Void => "Void".into(),
            VarType::Bool => "Bool".into(),
            VarType::U32 => "UInt32".into(),
            VarType::I32 => "Int32".into(),
            VarType::F32 => "Float32".into(),
            _ => unimplemented!(),
        }
    }
    pub fn stride(&self) -> usize {
        match self {
            VarType::Void => 0,
            VarType::Bool => bool::std140_size_static(),
            VarType::U32 => u32::std140_size_static(),
            VarType::I32 => i32::std140_size_static(),
            VarType::F32 => f32::std140_size_static(),
            _ => unimplemented!(),
        }
    }
    pub fn size(&self) -> usize {
        match self {
            VarType::Void => 0,
            VarType::Bool => bool::std140_size_static(),
            VarType::U32 => u32::std140_size_static(),
            VarType::I32 => i32::std140_size_static(),
            VarType::F32 => f32::std140_size_static(),
            _ => unimplemented!(),
        }
    }
    // #[allow(unused)]
    // pub fn from_rs<T: 'static>() -> Self {
    //     let ty_f32 = std::any::TypeId::of::<f32>();
    //     let ty_u32 = std::any::TypeId::of::<u32>();
    //     let ty_i32 = std::any::TypeId::of::<i32>();
    //     let ty_bool = std::any::TypeId::of::<bool>();
    //     match std::any::TypeId::of::<T>() {
    //         ty_f32 => Self::F32,
    //         ty_u32 => Self::U32,
    //         ty_i32 => Self::I32,
    //         ty_bool => Self::Bool,
    //         _ => unimplemented!(),
    //     }
    // }
    pub fn type_id(&self) -> TypeId {
        match self {
            VarType::Bool => TypeId::of::<bool>(),
            VarType::U32 => TypeId::of::<u32>(),
            VarType::I32 => TypeId::of::<i32>(),
            VarType::F32 => TypeId::of::<f32>(),
            _ => panic!("Error: {:?} type has no defined type id!", self),
        }
    }
    pub fn to_spirv(&self, b: &mut rspirv::dr::Builder) -> u32 {
        match self {
            VarType::Void => b.type_void(),
            VarType::Bool => b.type_bool(),
            VarType::U32 => b.type_int(32, 0),
            VarType::I32 => b.type_int(32, 1),
            VarType::F32 => b.type_float(32),
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
            VarType::U32 => Self::Int {
                width: 32,
                signedness: 0,
            },
            VarType::I32 => Self::Int {
                width: 32,
                signedness: 1,
            },
            VarType::F32 => Self::Float { width: 32 },
            _ => unimplemented!(),
        }
    }
}
