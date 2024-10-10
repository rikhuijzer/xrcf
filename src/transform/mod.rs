use crate::ir::Op;
use anyhow::Result;

mod func_to_llvm;

pub use func_to_llvm::ConvertFuncToLLVM;

pub trait Transform {
    fn transform(&self, op: &dyn Op) -> Result<()>;
}
