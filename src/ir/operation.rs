use crate::ir::Block;
use crate::ir::Region;
use crate::ir::Type;
use crate::ir::Value;
use crate::Attribute;
use std::default::Default;
use std::fmt::Display;
use std::fmt::Formatter;
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

/// Takes attributes
///
/// Note that MLIR distinguishes between Operation and Op.
/// Operation generically models all operations.
/// Op is an interface for more specific operations.
/// For example, `ConstantOp` does not take inputs and gives one output.
/// `ConstantOp` does also not specify fields since they are accessed
/// via a pointer to the `Operation`.
/// In MLIR, a specific Op can be casted from an Operation.
/// The operation also represents functions and modules.
///
/// Note that this type requires many methods. I guess this is a bit
/// inherent to the fact that an `Operation` aims to be very generic.
pub struct Operation {
    name: OperationName,
    operands: Arc<Vec<Value>>,
    attributes: Vec<Arc<dyn Attribute>>,
    result_types: Vec<Type>,
    regions: Vec<Pin<Box<Region>>>,
    parent_block: Option<Pin<Box<Block>>>,
}
impl Operation {
    pub fn new(
        name: OperationName,
        operands: Arc<Vec<Value>>,
        attributes: Vec<Arc<dyn Attribute>>,
        result_types: Vec<Type>,
        regions: Vec<Pin<Box<Region>>>,
        parent_block: Option<Pin<Box<Block>>>,
    ) -> Self {
        Self {
            name,
            operands,
            attributes,
            result_types,
            regions,
            parent_block,
        }
    }
    pub fn operands(&self) -> Arc<Vec<Value>> {
        self.operands.clone()
    }
    pub fn attributes(&self) -> Vec<Arc<dyn Attribute>> {
        self.attributes.clone()
    }
    pub fn result_types(&self) -> &Vec<Type> {
        &self.result_types
    }
    pub fn regions(&self) -> Vec<&Pin<Box<Region>>> {
        self.regions.iter().collect()
    }
    pub fn name(&self) -> String {
        self.name.name.clone()
    }
    pub fn set_name(&mut self, name: OperationName) -> &mut Self {
        self.name = name;
        self
    }
    pub fn set_operands(&mut self, operands: Arc<Vec<Value>>) -> &mut Self {
        self.operands = operands;
        self
    }
    pub fn set_attributes(&mut self, attributes: Vec<Arc<dyn Attribute>>) -> &mut Self {
        self.attributes = attributes;
        self
    }
    pub fn set_result_types(&mut self, result_types: Vec<Type>) -> &mut Self {
        self.result_types = result_types;
        self
    }
    pub fn set_regions(&mut self, regions: Vec<Pin<Box<Region>>>) -> &mut Self {
        self.regions = regions;
        self
    }
    fn display(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())?;
        for attribute in self.attributes.iter() {
            write!(f, " {}", attribute.display())?;
        }
        for region in self.regions() {
            write!(f, " {}", region)?;
        }
        Ok(())
    }
    /// Get the parent block (this is called `getBlock` in MLIR).
    fn parent_block(&self) -> Option<&Pin<Box<Block>>> {
        self.parent_block.as_ref()
    }
    // TODO: This is a temporary test.
    // TODO: This is a temporary test.
    pub fn rename(&mut self, name: String) {
        self.name = OperationName::new(name);
    }
}

impl Default for Operation {
    fn default() -> Self {
        Self {
            name: OperationName::new("".to_string()),
            operands: Arc::new(vec![]),
            attributes: vec![],
            result_types: vec![],
            regions: vec![],
            parent_block: None,
        }
    }
}

impl Display for Operation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f)
    }
}
