use crate::ir::Attributes;
use crate::ir::Block;
use crate::ir::Op;
use crate::ir::OpOperand;
use crate::ir::OpOperands;
use crate::ir::Region;
use crate::ir::Type;
use crate::ir::Types;
use crate::ir::Users;
use crate::ir::Values;
use crate::parser::Parser;
use crate::parser::ParserDispatch;
use crate::parser::TokenKind;
use anyhow::Result;
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
    pub fn new(name: String) -> Self {
        Self { name }
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

impl<T: ParserDispatch> Parser<T> {
    fn parse_operation_name(&mut self) -> Result<OperationName> {
        let identifier = self.expect(TokenKind::BareIdentifier)?;
        let name = OperationName::new(identifier.lexeme);
        Ok(name)
    }
    pub fn parse_operation_name_into<O: Op>(
        &mut self,
        operation: &mut Operation,
    ) -> Result<OperationName> {
        let name = self.parse_operation_name()?;
        assert!(name == O::operation_name());
        operation.set_name(name.clone());
        Ok(name)
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
    operands: OpOperands,
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
        Ok(())
    }
}

impl Operation {
    pub fn new(
        name: OperationName,
        arguments: Values,
        operands: OpOperands,
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
    pub fn operands(&self) -> OpOperands {
        self.operands.clone()
    }
    pub fn operand_types(&self) -> Types {
        let operands = self.operands.vec();
        let operands = operands.try_read().unwrap();
        let operand_types = operands
            .iter()
            .map(|o| o.try_read().unwrap().typ())
            .collect();
        Types::new(operand_types)
    }
    pub fn operand(&self, index: usize) -> Option<Arc<RwLock<OpOperand>>> {
        let operands = self.operands.vec();
        let operands = operands.try_read().unwrap();
        operands.get(index).cloned()
    }
    pub fn operands_mut(&mut self) -> &mut OpOperands {
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
    /// Get the single result type of the operation.
    ///
    /// Return `None` if the operation has fewer or more than 1 result.
    pub fn result_type(&self) -> Option<Arc<RwLock<dyn Type>>> {
        let result_types = self.result_types();
        let result_types = result_types.vec();
        let result_types = result_types.try_read().unwrap();
        if result_types.len() != 1 {
            return None;
        }
        let result_type = result_types.get(0).unwrap();
        Some(result_type.clone())
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
    /// Set the operand of the operation.
    ///
    /// This overrides any previously set operands.
    pub fn set_operand(&mut self, operand: Arc<RwLock<OpOperand>>) {
        let operands = vec![operand];
        self.operands = OpOperands::from_vec(operands);
    }
    pub fn set_operands(&mut self, operands: OpOperands) {
        self.operands = operands;
    }
    pub fn set_attributes(&mut self, attributes: Attributes) {
        self.attributes = attributes;
    }
    pub fn set_results(&mut self, results: Values) {
        self.results = results;
    }
    /// Set the result type of the operation.
    ///
    /// This overrides any previously set result types.
    pub fn set_result_type(&mut self, result_type: Arc<RwLock<dyn Type>>) {
        let result_types = vec![result_type];
        let result_types = Types::new(result_types);
        self.result_types = result_types;
    }
    /// Set the result types of the operation.
    ///
    /// This overrides any previously set result types. For overriding a single
    /// result type, use `set_result_type`.
    pub fn set_result_types(&mut self, result_types: Types) {
        self.result_types = result_types;
    }
    pub fn set_region(&mut self, region: Option<Arc<RwLock<Region>>>) {
        self.region = region;
    }
    pub fn set_parent(&mut self, parent: Option<Arc<RwLock<Block>>>) {
        self.parent = parent;
    }
    pub fn rename(&mut self, name: String) {
        self.name = OperationName::new(name);
    }
    pub fn users(&self) -> Users {
        let mut out = Vec::new();
        let results = self.results().vec();
        let results = results.try_read().unwrap();
        let defines_result = results.len() > 0;
        if !defines_result {
            return Users::HasNoOpResults;
        }
        for result in results.iter() {
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
    /// Display the results of the operation (e.g., `%0 = `).
    pub fn display_results(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let results = self.results().vec();
        let results = results.try_read().unwrap();
        if !results.is_empty() {
            write!(f, "{} = ", self.results())?;
        }
        Ok(())
    }
    pub fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        let spaces = crate::ir::spaces(indent);
        write!(f, "{spaces}")?;
        self.display_results(f)?;
        write!(f, "{}", self.name())?;
        let operands = self.operands();
        if !operands.vec().try_read().unwrap().is_empty() {
            write!(f, " {}", operands)?;
        }
        write!(f, "{}", self.attributes())?;
        let result_types = self.result_types();
        let result_types = result_types.vec();
        let result_types = result_types.try_read().unwrap();
        if !result_types.is_empty() {
            write!(f, " :")?;
            for result_type in result_types.iter() {
                write!(f, " {}", result_type.try_read().unwrap())?;
            }
        }
        display_region_inside_func(f, self, indent)
    }
}

impl Default for Operation {
    fn default() -> Self {
        Self {
            name: OperationName::new("".to_string()),
            arguments: Values::default(),
            operands: OpOperands::default(),
            attributes: Attributes::new(),
            results: Values::default(),
            result_types: Types::default(),
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
