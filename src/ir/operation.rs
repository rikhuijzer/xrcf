use crate::Attributes;
use anyhow::Result;
use crate::ir::Region;
use std::fmt::Display;

#[derive(Clone)]
pub struct OperationName {
    name: String, // TODO: Should be StringAttr,
}

impl OperationName {
    pub fn new(name: String) -> Self {
        Self { name }
    }
}

// Takes attributes
//
// Note that MLIR distinguishes between Operation and Op.
// Operation generically models all operations.
// Op is an interface for more specific operations.
// For example, `ConstantOp` does not take inputs and gives one output.
// `ConstantOp` does also not specify fields since they are accessed
// via a pointer to the `Operation`.
// In MLIR, a specific Op can be casted from an Operation.
// The operation also represents functions and modules.
#[derive(Clone)]
pub struct Operation {
    name: OperationName,
    regions: Vec<Region>,
    // operands: i64,
    // attributes: Attributes,
}
impl Operation {
    pub fn new(name: OperationName, regions: Vec<Region>) -> Self {
        Self { name, regions }
    }
    pub fn regions(&self) -> Vec<Region> {
        self.regions.to_vec()
    }
    pub fn name(&self) -> String {
        self.name.name.clone()
    }
    fn print(&self) {
        println!("{}", self.name());
    }
}

impl Display for Operation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())?;
        for region in self.regions() {
            write!(f, " {}", region)?;
        }
        Ok(())
    }
}

/// This is the trait that is implemented by all operations.
/// FuncOp, for example, will be implemented by various dialects.
/// Note that the parser will parse the tokens into an `Operation`
/// and MLIR would cast the `Operation` into a specific `Op` variant
/// such as `FuncOp`.
pub trait Op {
    fn from_operation(operation: Operation) -> Result<Self>
    where
        Self: Sized;
    fn operation(&self) -> Operation {
        self.operation()
    }
    fn name(&self) -> &'static str;
}