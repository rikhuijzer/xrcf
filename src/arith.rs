use crate::ir::attribute::IntegerAttr;
use crate::ir::operation::Operation;
use crate::ir::Op;
use crate::typ::IntegerType;
use crate::Dialect;
use anyhow::Result;
use std::pin::Pin;

struct Arith {}

struct ConstantOp {
    operation: Pin<Box<Operation>>,
    typ: IntegerType,
    value: IntegerAttr,
}

impl Op for ConstantOp {
    fn name() -> &'static str {
        "arith.contant"
    }
    fn from_operation(operation: Pin<Box<Operation>>) -> Result<Self> {
        todo!()
    }
    fn operation(&self) -> Pin<Box<Operation>> {
        self.operation.clone()
    }
}

struct Addi {
    operation: Pin<Box<Operation>>,
    lhs: IntegerAttr,
    rhs: IntegerAttr,
}

impl Op for Addi {
    fn name() -> &'static str {
        "arith.addi"
    }
    fn from_operation(operation: Pin<Box<Operation>>) -> Result<Self> {
        todo!()
    }
    fn operation(&self) -> Pin<Box<Operation>> {
        self.operation.clone()
    }

    // fn parse(input: &str) -> Option<Self> {
    // In MLIR this works by taking an OpAsmParser and parsing
    // the elements of the op.
    // Parsing tries to cast the elements to the expected types.
    // If all succeeds, the elements are parsed into the struct.
    // todo!()
    // }
}

// enum ArithOp {
//    Addi(Addi),
//}

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
