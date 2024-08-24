use crate::Dialect;
use crate::op::{Attribute, Attributes, Op};

struct Arith {}

struct Addi {}

impl Op for Addi {
    fn name(&self) -> &'static str {
        "arith.addi"
    }

    fn new(name: &'static str, attrs: impl Attributes) -> Self {
        Addi {}
    }

    fn verify(&self) -> bool {
        // TODO: Verify that operands are integerattrs
        false
    }
    fn parse(input: &str) -> Self {
        Addi {}
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