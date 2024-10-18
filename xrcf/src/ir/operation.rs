use crate::ir::Block;
use crate::ir::OpOperand;
use crate::ir::Region;
use crate::ir::Type;
use crate::ir::Users;
use crate::ir::Value;
use crate::Attribute;
use std::collections::HashMap;
use std::default::Default;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::RwLock;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct OperationName {
    name: String,
}

impl OperationName {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
        }
    }
    pub fn name(&self) -> String {
        self.name.clone()
    }
}

impl Display for OperationName {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.name.is_empty() {
            write!(f, "<unknown>")?;
        }
        write!(f, "{}", self.name)
    }
}

pub type Values = Arc<RwLock<Vec<Arc<RwLock<Value>>>>>;
pub type Operands = Arc<RwLock<Vec<Arc<RwLock<OpOperand>>>>>;
pub type Types = Arc<RwLock<Vec<Arc<RwLock<Type>>>>>;

#[derive(Clone)]
pub struct Attributes {
    map: Arc<RwLock<HashMap<String, Arc<dyn Attribute>>>>,
}

impl Attributes {
    pub fn new() -> Self {
        Self {
            map: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    pub fn map(&self) -> Arc<RwLock<HashMap<String, Arc<dyn Attribute>>>> {
        self.map.clone()
    }
    pub fn insert(&self, name: &str, attribute: Arc<dyn Attribute>) {
        self.map
            .write()
            .unwrap()
            .insert(name.to_string(), attribute);
    }
    pub fn deep_clone(&self) -> Self {
        let map = self.map.read().unwrap();
        let mut out = HashMap::new();
        for (name, attribute) in map.iter() {
            let attribute = attribute.clone();
            out.insert(name.to_string(), attribute);
        }
        let map = Arc::new(RwLock::new(out));
        Self { map }
    }
}

impl Display for Attributes {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for (name, attribute) in self.map.read().unwrap().iter() {
            if name == "value" {
                write!(f, " {attribute}")?;
            } else {
                write!(f, " {name} = {attribute}")?;
            }
        }
        Ok(())
    }
}

/// Note that MLIR distinguishes between Operation and Op.
/// Operation generically models all operations.
/// Op is an interface for more specific operations.
/// For example, `ConstantOp` does not take inputs and gives one output.
/// `ConstantOp` does also not specify fields apart from `operation` since
/// they are accessed via a pointer to the `Operation`.
/// In MLIR, a specific Op can be casted from an Operation.
/// The operation also represents functions and modules.
///
/// Note that this type requires many methods. I guess this is a bit
/// inherent to the fact that an `Operation` aims to be very generic.
#[derive(Clone)]
pub struct Operation {
    name: OperationName,
    /// Used by the `Func` trait implementers to store arguments.
    arguments: Values,
    operands: Operands,
    attributes: Attributes,
    /// Results can be `Values`, so either `BlockArgument` or `OpResult`.
    results: Values,
    result_types: Types,
    region: Option<Arc<RwLock<Region>>>,
    /// This is set after parsing because not all parents are known during
    /// parsing (for example, the parent of a top-level function will be a
    /// `ModuleOp` that is created after parsing of the `FuncOp`).
    parent: Option<Arc<RwLock<Block>>>,
}

pub fn display_region_inside_func(
    f: &mut Formatter<'_>,
    operation: &Operation,
    indent: i32,
) -> std::fmt::Result {
    let region = operation.region();
    if let Some(region) = region {
        let region = region.read().unwrap();
        if region.blocks().is_empty() {
            write!(f, "\n")
        } else {
            region.display(f, indent)
        }
    } else {
        write!(f, "\n")
    }
}

impl Operation {
    pub fn new(
        name: OperationName,
        arguments: Values,
        operands: Operands,
        attributes: Attributes,
        results: Values,
        result_types: Types,
        region: Option<Arc<RwLock<Region>>>,
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
    pub fn name(&self) -> OperationName {
        self.name.clone()
    }
    pub fn arguments(&self) -> Values {
        self.arguments.clone()
    }
    pub fn operands(&self) -> Operands {
        self.operands.clone()
    }
    pub fn operand(&self, index: usize) -> Arc<RwLock<OpOperand>> {
        self.operands.read().unwrap()[index].clone()
    }
    pub fn operands_mut(&mut self) -> &mut Operands {
        &mut self.operands
    }
    pub fn attributes(&self) -> Attributes {
        self.attributes.clone()
    }
    pub fn results(&self) -> Values {
        self.results.clone()
    }
    pub fn result_types(&self) -> Types {
        self.result_types.clone()
    }
    pub fn region(&self) -> Option<Arc<RwLock<Region>>> {
        self.region.clone()
    }
    /// Get the parent block (this is called `getBlock` in MLIR).
    pub fn parent(&self) -> Option<Arc<RwLock<Block>>> {
        self.parent.clone()
    }
    pub fn set_name(&mut self, name: OperationName) {
        self.name = name;
    }
    pub fn set_arguments(&mut self, arguments: Values) {
        self.arguments = arguments;
    }
    pub fn set_operands(&mut self, operands: Operands) {
        self.operands = operands;
    }
    pub fn set_attributes(&mut self, attributes: Attributes) {
        self.attributes = attributes;
    }
    pub fn set_results(&mut self, results: Values) {
        self.results = results;
    }
    pub fn set_result_types(&mut self, result_types: Types) {
        self.result_types = result_types;
    }
    pub fn set_region(&mut self, region: Option<Arc<RwLock<Region>>>) {
        self.region = region;
    }
    pub fn set_parent(&mut self, parent: Option<Arc<RwLock<Block>>>) {
        self.parent = parent;
    }
    pub fn rename(&mut self, name: &str) {
        self.name = OperationName::new(name);
    }
    pub fn users(&self) -> Users {
        let mut out = Vec::new();
        let defines_result = self.results().try_read().unwrap().len() > 0;
        if !defines_result {
            return Users::HasNoOpResults;
        }
        for result in self.results().try_read().unwrap().iter() {
            let result = result.try_read().unwrap();
            let result_users = result.users();
            match result_users {
                Users::OpOperands(users) => {
                    for usage in users.iter() {
                        out.push(usage.clone());
                    }
                }
                Users::HasNoOpResults => (),
            }
        }
        Users::OpOperands(out)
    }
    pub fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        let spaces = crate::ir::spaces(indent);
        write!(f, "{spaces}")?;
        if !self.results().read().unwrap().is_empty() {
            for result in self.results().read().unwrap().iter() {
                write!(f, "{}", result.read().unwrap())?;
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
                .map(|o| o.read().unwrap().name())
                .collect::<Vec<String>>()
                .join(", ");
            write!(f, " {}", joined)?;
        }
        write!(f, "{}", self.attributes())?;
        let result_types = self.result_types();
        let result_types = result_types.read().unwrap();
        if !result_types.is_empty() {
            write!(f, " :")?;
            for result_type in result_types.iter() {
                write!(f, " {}", result_type.read().unwrap())?;
            }
        }
        display_region_inside_func(f, self, indent)
    }
}

impl Default for Operation {
    fn default() -> Self {
        Self {
            name: OperationName::new(""),
            arguments: Arc::new(RwLock::new(vec![])),
            operands: Arc::new(RwLock::new(vec![])),
            attributes: Attributes::new(),
            results: Arc::new(RwLock::new(vec![])),
            result_types: Arc::new(RwLock::new(vec![])),
            region: None,
            parent: None,
        }
    }
}

impl Display for Operation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}
