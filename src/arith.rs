use crate::Dialect;
use crate::op::Attribute;
use crate::op::Attributes;
use crate::op::Op;

struct Arith {
}

struct Addi {
}

// Takes IntegerAttrs

impl Addi for Op {
    fn new(attrs: impl Attributes) -> Self {
        Op {
            name: "addi",
            attrs: attrs,
        }
    }
}

impl Dialect for Arith {

    fn name(&self) -> &'static str {
        "Arith"
    }

    fn description(&self) -> &'static str {
        "Arithmetic operations."
    }

    fn ops(&self) -> Vec<impl Op> {
        vec![]
    }
}
