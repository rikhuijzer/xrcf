use crate::Dialect;
use crate::op::{Attribute, Attributes, Op};

struct Arith {}

struct Add {}

impl Op for Add {
    fn name(&self) -> &'static str {
        "arith.add"
    }

    fn new(name: &'static str, attrs: impl Attributes) -> Self {
        Add {}
    }

    fn verify(&self) -> bool {
        true
    }
}

impl Dialect for Arith {
    fn name(&self) -> &'static str {
        "arith"
    }

    fn description(&self) -> &'static str {
        "Arithmetic operations for integers and floating-point values."
    }

    fn ops(&self) -> Vec<impl Op> {
        vec![Add {}]
    }
}
