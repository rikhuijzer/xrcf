//! LLVM dialect.
//!
//! This dialect is meant to hold operations that are related to the LLVM IR.
//! Note that operations in this dialect are not the same as operations in the
//! LLVM IR.

mod attribute;
mod op;
mod typ;

use crate::Dialect;

pub use op::AddOp;
pub use op::AllocaOp;
pub use op::BranchOp;
pub use op::CallOp;
pub use op::CondBranchOp;
pub use op::ConstantOp;
pub use op::FuncOp;
pub use op::GlobalOp;
pub use op::ReturnOp;
pub use op::StoreOp;
pub use typ::ArrayType;
pub use typ::FunctionType;
pub use typ::PointerType;
pub use typ::VariadicType;

pub struct LLVM {}

impl Dialect for LLVM {
    fn name(&self) -> &'static str {
        "llvm"
    }
    fn description(&self) -> &'static str {
        "Arithmetic dialect"
    }
}
