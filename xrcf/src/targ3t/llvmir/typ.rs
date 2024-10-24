use crate::ir::Type;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::RwLock;

#[derive(Clone)]
pub struct ArrayType {
    num_elements: u32,
    element_type: Arc<RwLock<dyn Type>>,
}

impl ArrayType {
    pub fn from_str(s: &str) -> Self {
        assert!(s.starts_with("["), "Expected {} to start with [", s);
        todo!("ArrayType not impl")
    }
}

impl Type for ArrayType {
    fn display(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let element_type = self.element_type.read().unwrap();
        write!(f, "[{} x {}]", self.num_elements, element_type)
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

#[derive(Clone)]
pub struct PointerType {}

impl PointerType {
    pub fn new() -> Self {
        Self {}
    }
    pub fn from_str(_s: &str) -> Self {
        Self {}
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