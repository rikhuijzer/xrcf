use crate::dialect::llvm::LLVM;
use crate::ir::IntegerType;
use crate::ir::Type;
use crate::ir::TypeParse;
use anyhow::Result;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::RwLock;

/// Represent an integer type such as i32 or i64.
///
/// Just like in LLVM, this does not include the sign bit since the sign does
/// not matter for 2s complement integer arithmetic.
#[derive(Clone)]
pub struct ArrayType {
    num_elements: u32,
    element_type: Arc<RwLock<dyn Type>>,
}

impl ArrayType {
    pub fn parse_str(s: &str) -> Self {
        assert!(s.starts_with("!llvm.array"));
        let s = s.strip_prefix("!llvm.array<").unwrap();
        let s = match s.strip_suffix(">") {
            Some(s) => s,
            None => {
                panic!("Failed to strip suffix from {}", s);
            }
        };
        let (num_elements, element_type) = s.split_once('x').unwrap();
        let num_elements = num_elements.parse::<u32>().unwrap();
        let element_type = IntegerType::from_str(element_type);
        let element_type = Arc::new(RwLock::new(element_type));
        Self {
            num_elements,
            element_type,
        }
    }
    /// Create an array type that can hold the given text.
    pub fn for_str(s: &str) -> Self {
        let text = s.to_string();
        let text = text.trim_matches('"');
        let num_elements = text.as_bytes().len() as u32;
        let element_type = IntegerType::from_str("i8");
        let element_type = Arc::new(RwLock::new(element_type));
        Self {
            num_elements,
            element_type,
        }
    }
    pub fn for_bytes(bytes: &Vec<u8>) -> Self {
        let num_elements = bytes.len() as u32;
        let element_type = IntegerType::from_str("i8");
        let element_type = Arc::new(RwLock::new(element_type));
        Self {
            num_elements,
            element_type,
        }
    }
    pub fn num_elements(&self) -> u32 {
        self.num_elements
    }
}

impl Type for ArrayType {
    fn display(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let element_type = self.element_type.read().unwrap();
        write!(f, "!llvm.array<{} x {}>", self.num_elements, element_type)
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
        write!(f, "!llvm.ptr")
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl TypeParse for LLVM {
    fn parse_type(src: &str) -> Result<Arc<RwLock<dyn Type>>> {
        if src.starts_with("!llvm.array") {
            return Ok(Arc::new(RwLock::new(ArrayType::parse_str(src))));
        }
        if src.starts_with("!llvm.ptr") {
            return Ok(Arc::new(RwLock::new(PointerType::from_str(src))));
        }
        todo!("Not yet implemented for {}", src)
    }
}
