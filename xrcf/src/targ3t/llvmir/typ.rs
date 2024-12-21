use crate::ir::Type;
use crate::ir::Types;
use crate::shared::Shared;
use crate::shared::SharedExt;
use std::fmt::Display;
use std::fmt::Formatter;

#[derive(Clone)]
pub struct ArrayType {
    num_elements: u32,
    element_type: Shared<dyn Type>,
}

impl std::str::FromStr for ArrayType {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        assert!(s.starts_with("["), "Expected {} to start with [", s);
        todo!("ArrayType not impl")
    }
}

impl Type for ArrayType {
    fn display(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{} x {}]", self.num_elements, self.element_type.rd())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Display for ArrayType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f)
    }
}

pub struct FunctionType {
    return_types: Types,
    arguments: Types,
}

impl FunctionType {
    pub fn new(return_types: Types, arguments: Types) -> Self {
        Self {
            return_types,
            arguments,
        }
    }
    pub fn return_types(&self) -> &Types {
        &self.return_types
    }
    pub fn arguments(&self) -> &Types {
        &self.arguments
    }
}

impl Type for FunctionType {
    fn display(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.arguments)
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Clone)]
pub struct PointerType {}

impl PointerType {
    pub fn new() -> Self {
        Self {}
    }
}

impl std::str::FromStr for PointerType {
    type Err = anyhow::Error;
    fn from_str(_s: &str) -> Result<Self, Self::Err> {
        Ok(Self {})
    }
}

impl Type for PointerType {
    fn display(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ptr")
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Display for PointerType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f)
    }
}
