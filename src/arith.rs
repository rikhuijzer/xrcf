use crate::op::Op;
use crate::attribute::IntegerAttr;
use crate::attribute::Attributes;
use crate::Dialect;

struct Arith {}

struct Addi {
    lhs: IntegerAttr,
    rhs: IntegerAttr,
}

impl Op for Addi {
    fn name(&self) -> &'static str {
        "arith.addi"
    }

    fn new(name: &'static str, attrs: Attributes) -> Self {
        todo!()
    }

    fn parse(input: &str) -> Option<Self> {
        // In MLIR this works by taking an OpAsmParser and parsing
        // the elements of the op.
        // Parsing tries to cast the elements to the expected types.
        // If all succeeds, the elements are parsed into the struct.
        todo!()
    }
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
