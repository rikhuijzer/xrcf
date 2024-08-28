use crate::ir::Block;
use crate::ir::Region;
use crate::Attribute;
use std::fmt::Display;
use std::pin::Pin;
use std::sync::Arc;

#[derive(Clone)]
pub struct OperationName {
    name: String, // TODO: Should be StringAttr,
}

impl OperationName {
    pub fn new(name: String) -> Self {
        Self { name }
    }
    pub fn name(&self) -> String {
        self.name.clone()
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
    attributes: Vec<Arc<dyn Attribute>>,
    regions: Vec<Region>,
    parent_block: Option<Pin<Box<Block>>>,
    // operands: i64,
}
impl Operation {
    pub fn new(
        name: OperationName,
        attributes: Vec<Arc<dyn Attribute>>,
        regions: Vec<Region>,
        parent_block: Option<Pin<Box<Block>>>,
    ) -> Self {
        Self {
            name,
            attributes,
            regions,
            parent_block,
        }
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
    /// Get the parent block (this is called `getBlock` in MLIR).
    fn parent_block(&self) -> Option<Pin<Box<Block>>> {
        self.parent_block.clone()
    }
    // TODO: This is a temporary test.
    pub fn rename(&mut self, name: String) {
        self.name = OperationName::new(name);
    }
}

impl Display for Operation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())?;
        for attribute in self.attributes.iter() {
            write!(f, " {}", attribute.print())?;
        }
        for region in self.regions() {
            write!(f, " {}", region)?;
        }
        Ok(())
    }
}
