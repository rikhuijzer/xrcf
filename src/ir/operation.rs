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
use std::sync::RwLock;

#[derive(Clone)]
pub struct OperationName {
    name: String,
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
    results: Vec<Value>,
    result_types: Vec<Type>,
    region: Arc<RwLock<Region>>,
    parent: Option<Pin<Box<Block>>>,
}

impl Operation {
    pub fn new(
        name: OperationName,
        operands: Arc<Vec<Value>>,
        attributes: Vec<Arc<dyn Attribute>>,
        results: Vec<Value>,
        result_types: Vec<Type>,
        region: Arc<RwLock<Region>>,
        parent: Option<Pin<Box<Block>>>,
    ) -> Self {
        Self {
            name,
            operands,
            attributes,
            results,
            result_types,
            region,
            parent,
        }
    }
    pub fn operands(&self) -> Arc<Vec<Value>> {
        self.operands.clone()
    }
    pub fn attributes(&self) -> Vec<Arc<dyn Attribute>> {
        self.attributes.clone()
    }
    pub fn results(&self) -> &Vec<Value> {
        &self.results
    }
    pub fn result_types(&self) -> &Vec<Type> {
        &self.result_types
    }
    pub fn region(&self) -> Arc<RwLock<Region>> {
        self.region.clone()
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
    pub fn set_results(&mut self, results: Vec<Value>) -> &mut Self {
        self.results = results;
        self
    }
    pub fn set_result_types(&mut self, result_types: Vec<Type>) -> &mut Self {
        self.result_types = result_types;
        self
    }
    pub fn set_region(&mut self, region: Arc<RwLock<Region>>) -> &mut Self {
        self.region = region.clone();
        self
    }
    /// Get the parent block (this is called `getBlock` in MLIR).
    fn parent(&self) -> Option<&Pin<Box<Block>>> {
        self.parent.as_ref()
    }
    pub fn rename(&mut self, name: String) {
        self.name = OperationName::new(name);
    }
    pub fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        let spaces = crate::ir::spaces(indent);
        write!(f, "{spaces}")?;
        if !self.results().is_empty() {
            for result in self.results().iter() {
                write!(f, "{}", result)?;
            }
            write!(f, " = ")?;
        }
        write!(f, "{}", self.name())?;
        if !self.operands().is_empty() {
            let joined = self
                .operands()
                .iter()
                .map(|o| o.to_string())
                .collect::<Vec<String>>()
                .join(", ");
            write!(f, " {}", joined)?;
        }
        for attribute in self.attributes.iter() {
            write!(f, " {}", attribute)?;
        }
        if !self.result_types.is_empty() {
            write!(f, " :")?;
            for result_type in self.result_types.iter() {
                write!(f, " {}", result_type)?;
            }
        }
        let region = self.region();
        let region = region.read().unwrap();
        if region.is_empty() {
            write!(f, "\n")?;
        } else {
            region.display(f, indent)?;
        }
        Ok(())
    }
}

impl Default for Operation {
    fn default() -> Self {
        Self {
            name: OperationName::new("".to_string()),
            operands: Arc::new(vec![]),
            attributes: vec![],
            results: vec![],
            result_types: vec![],
            region: Arc::new(RwLock::new(Region::default())),
            parent: None,
        }
    }
}

impl Display for Operation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}
