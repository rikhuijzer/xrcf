//! LLVM IR dialect.
//!
//! This dialect holds operations that when printed are valid LLVM IR.

mod op;
mod typ;

use crate::Dialect;

pub use op::AddOp;
pub use op::AllocaOp;
pub use op::BranchOp;
pub use op::CallOp;
pub use op::FuncOp;
pub use op::ModuleOp;
pub use op::PhiOp;
pub use op::ReturnOp;
pub use op::StoreOp;
pub use typ::ArrayType;
pub use typ::FunctionType;
pub use typ::PointerType;

pub struct LLVMIR {}

impl Dialect for LLVMIR {
    fn name(&self) -> &'static str {
        "llvmir"
    }
    fn description(&self) -> &'static str {
        "LLVM IR dialect"
    }
}
