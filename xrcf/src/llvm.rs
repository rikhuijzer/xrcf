/// Generate LLVM IR.
use crate::op::Op;
use crate::attribute::IntegerAttr;
use crate::attribute::Attributes;
use crate::Dialect;

struct LLVM {}

struct Addi {
    typ: IntegerType,
    lhs: IntegerAttr,
    rhs: IntegerAttr,
}

impl Op for Addi {
    fn name(&self) -> &'static str {
        "add"
    }
}

impl Dialect for LLVM {
    fn name(&self) -> &'static str {
        "llvm"
    }

    fn description(&self) -> &'static str {
        "LLVM dialect."
    }
}
