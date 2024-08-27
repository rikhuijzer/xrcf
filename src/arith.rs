use crate::ir::operation::Operation;
use crate::attribute::IntegerAttr;
use crate::typ::IntegerType;
use crate::Dialect;
use crate::ir::operation::Op;

struct Arith {}

struct ConstantOp {
    typ: IntegerType,
    value: IntegerAttr,
}

impl Op for ConstantOp {
    fn name(&self) -> &'static str {
        "arith.contant"
    }
}

struct Addi {
    lhs: IntegerAttr,
    rhs: IntegerAttr,
}

impl Op for Addi {
    fn name(&self) -> &'static str {
        "arith.addi"
    }

    // fn parse(input: &str) -> Option<Self> {
        // In MLIR this works by taking an OpAsmParser and parsing
        // the elements of the op.
        // Parsing tries to cast the elements to the expected types.
        // If all succeeds, the elements are parsed into the struct.
        // todo!()
    // }
}

enum ArithOp {
    Addi(Addi),
}

impl Dialect for Arith {
    fn name(&self) -> &'static str {
        "arith"
    }

    fn description(&self) -> &'static str {
        "Arithmetic operations."
    }

    // Probably we don't want to have a global obs state but instead
    // have some differrent implementations for common functions.
    // fn ops(&self) -> 
    // }
}
