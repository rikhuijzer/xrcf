use crate::dialect::llvm::LLVM;
use crate::ir::IntegerType;
use crate::ir::Type;
use crate::ir::TypeParse;
use crate::ir::Types;
use crate::shared::Shared;
use crate::shared::SharedExt;
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
    element_type: Shared<dyn Type>,
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
        let element_type = Shared::new(element_type.into());
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
        let element_type = Shared::new(element_type.into());
        Self {
            num_elements,
            element_type,
        }
    }
    pub fn for_bytes(bytes: &Vec<u8>) -> Self {
        let num_elements = bytes.len() as u32;
        let element_type = IntegerType::from_str("i8");
        let element_type = Shared::new(element_type.into());
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
        write!(
            f,
            "!llvm.array<{} x {}>",
            self.num_elements,
            self.element_type.rd()
        )
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
    /// Parse `!llvm.func<i32(i32, ...)>`.
    pub fn from_str(s: &str) -> Self {
        assert!(s.starts_with("!llvm.func<"));
        let s = s.strip_prefix("!llvm.func<").unwrap();
        let (return_types, arguments) = s.split_once('(').unwrap();

        let return_type = IntegerType::from_str(return_types.trim());
        let return_type = Shared::new(return_type.into());
        let return_types = Types::from_vec(vec![return_type]);

        let arguments_str = arguments.trim_end_matches(")>");
        let arguments_str = arguments_str.trim_start_matches('(');
        let mut arguments = vec![];
        for argument in arguments_str.split(',') {
            let argument = LLVM::parse_type(argument.trim()).unwrap();
            arguments.push(argument);
        }
        let arguments = Types::from_vec(arguments);

        Self {
            return_types,
            arguments,
        }
    }
}

impl Type for FunctionType {
    fn display(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "!llvm.func<")?;
        write!(f, "{}", self.return_types)?;
        write!(f, " (")?;
        write!(f, "{}", self.arguments)?;
        write!(f, ")>")
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

#[derive(Clone)]
pub struct VariadicType {}

impl VariadicType {
    pub fn new() -> Self {
        Self {}
    }
    pub fn from_str(s: &str) -> Self {
        assert!(s == "...", "Expected '...', but got {s}");
        Self {}
    }
}

impl Type for VariadicType {
    fn display(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "...")
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl TypeParse for LLVM {
    fn parse_type(src: &str) -> Result<Arc<RwLock<dyn Type>>> {
        if src.starts_with("!llvm.array") {
            return Ok(Shared::new(ArrayType::parse_str(src).into()));
        }
        if src.starts_with("!llvm.func") {
            return Ok(Shared::new(FunctionType::from_str(src).into()));
        }
        if src.starts_with("!llvm.ptr") {
            return Ok(Shared::new(PointerType::from_str(src).into()));
        }
        if src.starts_with("ptr") {
            return Ok(Shared::new(PointerType::from_str(src).into()));
        }
        if src == "..." {
            return Ok(Shared::new(VariadicType::from_str(src).into()));
        }
        todo!("Not yet implemented for {}", src)
    }
}
