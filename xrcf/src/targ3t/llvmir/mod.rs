mod op;
mod typ;

use crate::Dialect;

pub use op::AddOp;
pub use op::AllocaOp;
pub use op::CallOp;
pub use op::FuncOp;
pub use op::ModuleOp;
pub use op::ReturnOp;
pub use op::StoreOp;
pub use typ::ArrayType;
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
