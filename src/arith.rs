use crate::Dialect;

struct Arith {
    name: &'static str,
    description: &'static str,
}

impl Dialect for Arith {
    fn new(name: &'static str, description: &'static str) -> Self {
        Arith { name, description }
    }

    fn name(&self) -> &'static str {
        "Arith"
    }

    fn description(&self) -> &'static str {
        "Arithmetic operations."
    }
}
