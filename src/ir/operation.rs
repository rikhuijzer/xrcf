use crate::ir::Block;
use crate::ir::BlockArgument;
use crate::ir::OpOperand;
use crate::ir::Region;
use crate::ir::Type;
use crate::ir::Value;
use crate::Attribute;
use std::collections::HashMap;
use std::default::Default;
use std::fmt::Display;
use std::fmt::Formatter;
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

pub type OperationArguments = Arc<RwLock<Vec<Arc<RwLock<BlockArgument>>>>>;
pub type OperationOperands = Arc<RwLock<Vec<Arc<RwLock<OpOperand>>>>>;
pub type OperationAttributes = Arc<RwLock<HashMap<String, Arc<dyn Attribute>>>>;

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
    /// Used by `FuncOp` to store its arguments.
    arguments: OperationArguments,
    operands: OperationOperands,
    attributes: OperationAttributes,
    /// Results can be `Values`, so either `BlockArgument` or `OpResult`.
    results: Vec<Arc<Value>>,
    result_types: Vec<Type>,
    region: Arc<RwLock<Option<Region>>>,
    /// This is set after parsing because not all parents are known during
    /// parsing (for example, the parent of a top-level function will be a
    /// `ModuleOp` that is created after parsing of the `FuncOp`).
    parent: Option<Arc<RwLock<Block>>>,
}

impl Operation {
    pub fn new(
        name: OperationName,
        arguments: OperationArguments,
        operands: OperationOperands,
        attributes: OperationAttributes,
        results: Vec<Arc<Value>>,
        result_types: Vec<Type>,
        region: Arc<RwLock<Option<Region>>>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Self {
        Self {
            name,
            arguments,
            operands,
            attributes,
            results,
            result_types,
            region,
            parent,
        }
    }
    pub fn name(&self) -> String {
        self.name.name.clone()
    }
    pub fn arguments(&self) -> OperationArguments {
        self.arguments.clone()
    }
    pub fn operands(&self) -> OperationOperands {
        self.operands.clone()
    }
    pub fn operands_mut(&mut self) -> &mut OperationOperands {
        &mut self.operands
    }
    pub fn attributes(&self) -> OperationAttributes {
        self.attributes.clone()
    }
    pub fn results(&self) -> &Vec<Arc<Value>> {
        &self.results
    }
    pub fn result_types(&self) -> &Vec<Type> {
        &self.result_types
    }
    pub fn region(&self) -> Arc<RwLock<Option<Region>>> {
        self.region.clone()
    }
    /// Get the parent block (this is called `getBlock` in MLIR).
    fn parent(&self) -> Option<Arc<RwLock<Block>>> {
        self.parent.clone()
    }
    pub fn set_name(&mut self, name: OperationName) {
        self.name = name;
    }
    pub fn set_arguments(&mut self, arguments: OperationArguments) {
        self.arguments = arguments;
    }
    pub fn set_operands(&mut self, operands: OperationOperands) {
        self.operands = operands;
    }
    pub fn set_attributes(&mut self, attributes: OperationAttributes) {
        self.attributes = attributes;
    }
    pub fn set_results(&mut self, results: Vec<Arc<Value>>) {
        self.results = results;
    }
    pub fn set_result_types(&mut self, result_types: Vec<Type>) {
        self.result_types = result_types;
    }
    pub fn set_region(&mut self, region: Arc<RwLock<Option<Region>>>) {
        self.region = region;
    }
    pub fn set_parent(&mut self, parent: Option<Arc<RwLock<Block>>>) {
        self.parent = parent;
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
        if !self.operands().read().unwrap().is_empty() {
            let joined = self
                .operands()
                .read()
                .unwrap()
                .iter()
                .map(|o| o.read().unwrap().operand_name().to_string())
                .collect::<Vec<String>>()
                .join(", ");
            write!(f, " {}", joined)?;
        }
        for (name, attribute) in self.attributes().read().unwrap().iter() {
            write!(f, "{} = {}", name, attribute)?;
        }
        if !self.result_types.is_empty() {
            write!(f, " :")?;
            for result_type in self.result_types.iter() {
                write!(f, " {}", result_type)?;
            }
        }
        let region = self.region();
        if let Some(region) = region.read().unwrap().as_ref() {
            if region.blocks().is_empty() {
                write!(f, "\n")?;
            } else {
                region.display(f, indent)?;
            }
        } else {
            write!(f, "\n")?;
        }
        Ok(())
    }
}

impl Default for Operation {
    fn default() -> Self {
        Self {
            name: OperationName::new("".to_string()),
            arguments: Arc::new(RwLock::new(vec![])),
            operands: Arc::new(RwLock::new(vec![])),
            attributes: Arc::new(RwLock::new(HashMap::new())),
            results: vec![],
            result_types: vec![],
            region: Arc::new(RwLock::new(None)),
            parent: None,
        }
    }
}

impl Display for Operation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}
