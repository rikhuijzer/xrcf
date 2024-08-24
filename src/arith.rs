use crate::op::{Attribute, Attributes, Op};
use crate::Dialect;

struct Arith {}

struct Addi {
    lhs: IntegerAttr,
    rhs: IntegerAttr,
}

trait Type {}

/// Represents an integer type such as i32 or i64.
/// This does not include the sign bit like in LLVM since
/// it doesn't matter for 2s complement integer arithmetic.
struct IntegerType {
    num_bits: u64,
}

impl Type for IntegerType {}

/// Arbitrary precision integer.
struct APInt {
    num_bits: u64,
    value: u64,
    is_signed: bool,
}

/// An attribute containing an integer value.
struct IntegerAttr {
    // The type of the integer like the precision.
    typ: IntegerType,
    // An arbitrary precision integer value.
    value: APInt,
}

impl Op for Addi {
    fn name(&self) -> &'static str {
        "arith.addi"
    }

    fn new(name: &'static str, attrs: impl Attributes) -> Self {
        Addi {}
    }

    fn parse(input: &str) -> Option<Self> {
        // In MLIR this works by taking an OpAsmParser and parsing
        // the elements of the op.
        // Parsing tries to cast the elements to the expected types.
        // If all succeeds, the elements are parsed into the struct.
        Some(Addi {
            lhs: IntegerAttr {
                typ: Type { name: "i32" },
                value: APInt { value: 1 },
            },
            rhs: IntegerAttr {
                typ: Type { name: "i32" },
                value: APInt { value: 2 },
            },
        })
    }
}

impl Dialect for Arith {
    fn name(&self) -> &'static str {
        "arith"
    }

    fn description(&self) -> &'static str {
        "Arithmetic operations."
    }

    fn ops(&self) -> Vec<impl Op> {
        vec![Addi {}]
    }
}
