mod attribute;
mod op;
mod typ;

use crate::Dialect;

pub use op::AddOp;
pub use op::AllocaOp;
pub use op::CallOp;
pub use op::ConstantOp;
pub use op::FuncOp;
pub use op::GlobalOp;
pub use op::ReturnOp;
pub use op::StoreOp;
pub use typ::ArrayType;
pub use typ::PointerType;

pub struct LLVM {}

impl Dialect for LLVM {
    fn name(&self) -> &'static str {
        "llvm"
    }
    fn description(&self) -> &'static str {
        "Arithmetic dialect"
    }
}
