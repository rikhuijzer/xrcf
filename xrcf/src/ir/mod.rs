//! Intermediate representation (IR) for the compiler.
//!
//! These data structures are used as the basis for the compiler.
//! For example, this module some core types such as [Operation] and [Op].

mod attribute;
mod block;
mod module;
mod op;
mod op_operand;
mod operation;
mod region;
mod typ;
mod value;

pub use attribute::AnyAttr;
pub use attribute::Attribute;
pub use attribute::Attributes;
pub use attribute::BooleanAttr;
pub use attribute::IntegerAttr;
pub use attribute::StringAttr;
pub use block::Block;
pub use block::GuardedBlock;
pub use block::UnsetBlock;
pub use module::ModuleOp;
pub use op::GuardedOp;
pub use op::Op;
pub use op::UnsetOp;
pub use op_operand::GuardedOpOperand;
pub use op_operand::OpOperand;
pub use op_operand::OpOperands;
pub use operation::display_region_inside_func;
pub use operation::GuardedOperation;
pub use operation::Operation;
pub use operation::OperationName;
pub use operation::RenameBareToPercent;
pub use operation::VariableRenamer;
pub use region::GuardedRegion;
pub use region::Region;
pub use typ::APInt;
pub use typ::AnyType;
pub use typ::IntegerType;
pub use typ::StringType;
pub use typ::Type;
pub use typ::TypeConvert;
pub use typ::TypeParse;
pub use typ::Types;
pub use value::AnonymousResult;
pub use value::BlockArgument;
pub use value::BlockArgumentName;
pub use value::BlockDest;
pub use value::BlockLabel;
pub use value::Constant;
pub use value::GuardedValue;
pub use value::OpResult;
pub use value::UnsetOpResult;
pub use value::UnsetOpResults;
pub use value::Users;
pub use value::Value;
pub use value::Values;

pub fn spaces(indent: i32) -> String {
    "  ".repeat(indent as usize)
}

/// Convert a string to a vector of bytes while handling LLVM escape sequences.
pub fn llvm_string_to_bytes(src: &str) -> Vec<u8> {
    let src = src.to_string();
    let src = src.replace("\\00", "\0");
    let src = src.replace("\\0A", "\n");
    let mut out: Vec<u8> = vec![];
    let chars = src.as_bytes();
    out.extend_from_slice(chars);
    out
}

#[test]
fn test_llvm_string_to_bytes() {
    let src = "hello\n";
    let bytes = llvm_string_to_bytes(src);
    assert_eq!(bytes.len(), 6);
    assert_eq!(bytes.last(), Some(&10));

    let src = "hello\n\\00";
    let bytes = llvm_string_to_bytes(src);
    assert_eq!(bytes.len(), 7);
    assert_eq!(bytes.get(5), Some(&10));
    assert_eq!(bytes.last(), Some(&0));
}

/// Convert a vector of bytes to a string while handling LLVM escape sequences.
///
// Explicitly print the null byte too or else the vector length will not match
// the text that is shown.

// A backslash with two hex characters defines the byte in LLVM IR in hex, so
// \00 is the null byte in LLVM IR.
pub fn bytes_to_llvm_string(bytes: &[u8]) -> String {
    let src = String::from_utf8(bytes.to_vec()).unwrap();
    src.replace("\0", "\\00").replace("\n", "\\0A")
}

#[test]
fn test_bytes_to_llvm_string() {
    let bytes = vec![104, 101, 108, 108, 111, 10, 0];
    let src = bytes_to_llvm_string(&bytes);
    assert_eq!(src, "hello\\0A\\00");
}

pub fn escape(src: &str) -> String {
    let src = src.replace("\n", "\\n");
    src
}

pub fn unescape(src: &str) -> String {
    let src = src.replace("\\n", "\n");
    src
}

/// Generate a new name as part of printing.
///
/// Assumes `used_names` contains only names that are defined before the current
/// `own_name` and is ordered by order of occurrence.
pub fn generate_new_name(used_names: Vec<String>, prefix: &str) -> String {
    let mut new_index: i32 = -1;
    for name in used_names {
        if name.starts_with(prefix) {
            let name = name.trim_start_matches(prefix);
            if let Ok(num) = name.parse::<i32>() {
                // Ensure new_name is greater than any used name.
                // This is required by LLVM.
                new_index = new_index.max(num);
            }
        }
    }
    new_index += 1;
    format!("{prefix}{new_index}")
}

#[test]
fn test_generate_new_name() {
    let used_names = vec!["arg0", "arg1", "arg2"];
    let used_names = used_names.iter().map(|s| s.to_string()).collect();
    let prefix = "arg";
    let new_name = generate_new_name(used_names, prefix);
    assert_eq!(new_name, "arg3");
}
