use crate::ir::AnonymousResult;
use crate::ir::Attributes;
use crate::ir::Block;
use crate::ir::BlockArgumentName;
use crate::ir::GuardedBlock;
use crate::ir::Op;
use crate::ir::OpOperand;
use crate::ir::OpOperands;
use crate::ir::OpResult;
use crate::ir::Region;
use crate::ir::Type;
use crate::ir::Types;
use crate::ir::UnsetOpResult;
use crate::ir::Users;
use crate::ir::Value;
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

/// A generic representation of an operation.
///
/// An [Operation] generically models all operations and is wrapped by an [Op].
/// The benefit of this is that [Operation] can contain many generic fields and
/// methods that are useful for most operations. At the same time, more specific
/// data and methods can be stored in the [Op].
///
/// For example, a very simple operation is `arith.addi`:
/// ```mlir
/// %x = arith.addi %a, %b : i32
/// ```
/// This operation has an operation name (`arith.addi`), two operands (`%a` and
/// `%b`) and one result (`%x`). These can all be stored in fields of
/// [Operation]. Furthermore, some helper functions such as
/// `parser.parse_op_operands_into` can take a parser and parse the operands
/// straight into the [Operation]. This makes it easier to write parsers.
///
/// Also, [Operation] has a default printer that can correctly print most simple
/// operations.
///
/// An operation that needs more specific fields is, for example, `func.func`:
/// ```mlir
/// func.func @some_name() {
///     return
/// }
/// ```
/// Here, [Op] contains a field `identifier` that contains "@some_name".
#[derive(Clone)]
pub struct Operation {
    name: OperationName,
    /// Used by the `Func` trait implementers to store arguments.
    arguments: Values,
    operands: OpOperands,
    attributes: Attributes,
    /// Results are [Value]s, so either [BlockArgument] or [OpResult].
    results: Values,
    region: Option<Arc<RwLock<Region>>>,
    /// This is set after parsing because not all parents are known during
    /// parsing (for example, the parent of a top-level function will be a
    /// `ModuleOp` that is created after parsing of the `FuncOp`).
    parent: Option<Arc<RwLock<Block>>>,
}

/// Two operations are equal if they point to the same object.
///
/// This is used for things liking finding `successors`.
impl PartialEq for Operation {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

pub fn display_region_inside_func(
    f: &mut Formatter<'_>,
    operation: &Operation,
    indent: i32,
) -> std::fmt::Result {
    let region = operation.region();
    if let Some(region) = region {
        let region = region.try_read().unwrap();
        let blocks = region.blocks();
        let blocks = blocks.vec();
        let blocks = blocks.try_read().unwrap();
        if blocks.is_empty() {
            write!(f, "\n")
        } else {
            region.display(f, indent)
        }
    } else {
        Ok(())
    }
}

/// Set the value at `index` or grow the vector by one if `index` equals the
/// length of the vector.
fn set_or_grow_by_one<T>(vec: &mut Vec<T>, index: usize, value: T) {
    if index < vec.len() {
        vec[index] = value;
    } else if index == vec.len() {
        vec.push(value);
    } else {
        panic!(
            "tried to set value at index {} but length is {}",
            index,
            vec.len()
        );
    }
}

pub trait VariableRenamer {
    fn rename(&self, name: &str) -> String;
}

/// Rename bare variables `x` to percent variables `%x`.
///
/// This step is necessary when going from programming languages like Python to
/// MLIR. This object can be passed to [Operation::rename_variables] during a
/// conversion pass.
pub struct RenameBareToPercent;

impl VariableRenamer for RenameBareToPercent {
    fn rename(&self, name: &str) -> String {
        assert!(!name.starts_with('%'));
        format!("%{}", name)
    }
}

impl Operation {
    pub fn new(
        name: OperationName,
        arguments: Values,
        operands: OpOperands,
        attributes: Attributes,
        results: Values,
        region: Option<Arc<RwLock<Region>>>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Self {
        Self {
            name,
            arguments,
            operands,
            attributes,
            results,
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
    pub fn blocks(&self) -> Vec<Arc<RwLock<Block>>> {
        let region = self.region();
        let region = region.expect("expected region");
        let region = region.try_read().unwrap();
        let blocks = region.blocks();
        let blocks = blocks.vec();
        let blocks = blocks.try_read().unwrap();
        blocks.to_vec()
    }
    pub fn operand_types(&self) -> Types {
        let operands = self.operands.vec();
        let operands = operands.try_read().unwrap();
        let operand_types = operands
            .iter()
            .map(|o| o.try_read().unwrap().typ().unwrap())
            .collect();
        Types::from_vec(operand_types)
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
    pub fn result(&self, index: usize) -> Option<Arc<RwLock<Value>>> {
        let results = self.results.vec();
        let results = results.try_read().unwrap();
        results.get(index).cloned()
    }
    /// Get the single result type of the operation.
    ///
    /// Return `None` if `results.len() =! 1`.
    pub fn result_type(&self, index: usize) -> Option<Arc<RwLock<dyn Type>>> {
        let results = self.results();
        let results = results.vec();
        let results = results.try_read().unwrap();
        let result = results.get(index).unwrap();
        let result = result.try_read().unwrap();
        Some(result.typ().unwrap())
    }
    pub fn result_names(&self) -> Vec<String> {
        let results = self.results();
        let results = results.vec();
        let results = results.try_read().unwrap();
        let mut result_names = vec![];
        for result in results.iter() {
            let result = result.try_read().unwrap();
            let name = match &*result {
                Value::BlockArgument(arg) => {
                    let name = arg.name();
                    let name = name.try_read().unwrap();
                    match &*name {
                        BlockArgumentName::Anonymous => continue,
                        BlockArgumentName::Name(name) => name.to_string(),
                        BlockArgumentName::Unset => continue,
                    }
                }
                Value::BlockLabel(label) => label.name(),
                Value::Constant(_) => continue,
                Value::FuncResult(_) => continue,
                Value::OpResult(res) => {
                    let name = res.name();
                    let name = name.try_read().unwrap();
                    match &*name {
                        Some(name) => name.to_string(),
                        None => continue,
                    }
                }
                Value::Variadic => continue,
            };
            result_names.push(name);
        }
        result_names
    }
    pub fn region(&self) -> Option<Arc<RwLock<Region>>> {
        self.region.clone()
    }
    /// Return the parent block (this is called `getBlock` in MLIR).
    pub fn parent(&self) -> Option<Arc<RwLock<Block>>> {
        self.parent.clone()
    }
    pub fn parent_op(&self) -> Option<Arc<RwLock<dyn Op>>> {
        let parent = match self.parent() {
            Some(parent) => parent,
            None => return None,
        };
        let parent = parent.try_read().unwrap();
        let parent = parent.parent();
        let parent = match parent {
            Some(parent) => parent,
            None => return None,
        };
        let parent = parent.try_read().unwrap();
        let parent = parent.parent();
        let parent = match parent {
            Some(parent) => parent,
            None => return None,
        };
        Some(parent)
    }
    pub fn rename_variables(&self, renamer: &dyn VariableRenamer) -> Result<()> {
        let results = self.results();
        results.rename_variables(renamer)
    }
    pub fn set_name(&mut self, name: OperationName) {
        self.name = name;
    }
    pub fn set_arguments(&mut self, arguments: Values) {
        self.arguments = arguments;
    }
    pub fn set_argument(&mut self, index: usize, argument: Arc<RwLock<Value>>) {
        let arguments = self.arguments.vec();
        let mut arguments = arguments.try_write().unwrap();
        set_or_grow_by_one(&mut arguments, index, argument);
    }
    pub fn set_operand(&mut self, index: usize, operand: Arc<RwLock<OpOperand>>) {
        let operands = self.operands.vec();
        let mut operands = operands.try_write().unwrap();
        set_or_grow_by_one(&mut operands, index, operand);
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
    /// Update the result type for the operation.
    ///
    /// Panics if `results.len() != 1`.
    pub fn set_result_type(&self, index: usize, result_type: Arc<RwLock<dyn Type>>) -> Result<()> {
        let results = self.results();
        let results = results.vec();
        let results = results.try_read().unwrap();
        let result = results.get(index).unwrap();
        let mut result = result.try_write().unwrap();
        result.set_type(result_type);
        Ok(())
    }
    /// Set the results (and types) of the operation to [AnonymousResult]s.
    pub fn set_anonymous_results(&self, result_types: Vec<Arc<RwLock<dyn Type>>>) -> Result<()> {
        let results = self.results();
        let results = results.vec();
        let mut results = results.try_write().unwrap();
        for result_type in result_types.iter() {
            let func_result = AnonymousResult::new(result_type.clone());
            let value = Value::FuncResult(func_result);
            let value = Arc::new(RwLock::new(value));
            results.push(value);
        }
        Ok(())
    }
    /// Set the result (and type) of the operation to [AnonymousResult].
    pub fn set_anonymous_result(&mut self, result_type: Arc<RwLock<dyn Type>>) -> Result<()> {
        let result_types = vec![result_type];
        self.set_anonymous_results(result_types)
    }
    pub fn predecessors(&self) -> Vec<Arc<RwLock<dyn Op>>> {
        let parent = self.parent();
        let parent = parent.expect("Expected parent");
        let index = match parent.index_of(self) {
            Some(index) => index,
            None => {
                panic!(
                    "Expected index. Is the parent set correctly for the following op?\n{}",
                    self
                );
            }
        };
        let ops = parent.ops();
        let ops = ops.try_read().unwrap();
        ops[..index].to_vec()
    }
    pub fn successors(&self) -> Vec<Arc<RwLock<dyn Op>>> {
        let parent = self.parent();
        let parent = parent.expect("Expected parent");
        let index = match parent.index_of(self) {
            Some(index) => index,
            None => {
                panic!(
                    "Expected index. Is the parent set correctly for the following op?\n{}",
                    self
                );
            }
        };
        let ops = parent.ops();
        let ops = ops.try_read().unwrap();
        ops[index + 1..].to_vec()
    }
    pub fn update_result_types(&mut self, result_types: Vec<Arc<RwLock<dyn Type>>>) -> Result<()> {
        let mut results = self.results();
        if results.vec().try_read().unwrap().is_empty() {
            return Err(anyhow::anyhow!("Expected results to have been set"));
        }
        results.update_types(result_types)?;
        Ok(())
    }
    /// Add a new op result with given name.
    pub fn add_new_op_result(&self, name: &str, typ: Arc<RwLock<dyn Type>>) -> UnsetOpResult {
        let results = self.results();
        let vec = results.vec();
        let mut vec = vec.try_write().unwrap();

        let mut result = OpResult::default();
        result.set_name(name);
        result.set_typ(typ);
        let op_result = Value::OpResult(result);
        let op_result = Arc::new(RwLock::new(op_result));
        vec.push(op_result.clone());
        UnsetOpResult::new(op_result)
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
    pub fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        let spaces = crate::ir::spaces(indent);
        write!(f, "{spaces}")?;
        if !self.results().is_empty() {
            write!(f, "{} = ", self.results())?;
        }
        write!(f, "{}", self.name())?;
        let operands = self.operands();
        if !operands.vec().try_read().unwrap().is_empty() {
            write!(f, " {}", operands)?;
        }
        write!(f, "{}", self.attributes())?;
        let result_types = self.results().types();
        if !result_types.vec().is_empty() {
            write!(f, " :")?;
            for result_type in result_types.vec().iter() {
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

pub trait GuardedOperation {
    fn arguments(&self) -> Values;
    fn attributes(&self) -> Attributes;
    fn blocks(&self) -> Vec<Arc<RwLock<Block>>>;
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result;
    fn name(&self) -> OperationName;
    fn operand(&self, index: usize) -> Option<Arc<RwLock<OpOperand>>>;
    fn operands(&self) -> OpOperands;
    fn parent(&self) -> Option<Arc<RwLock<Block>>>;
    fn predecessors(&self) -> Vec<Arc<RwLock<dyn Op>>>;
    fn region(&self) -> Option<Arc<RwLock<Region>>>;
    fn rename_variables(&self, renamer: &dyn VariableRenamer) -> Result<()>;
    fn result(&self, index: usize) -> Option<Arc<RwLock<Value>>>;
    fn result_names(&self) -> Vec<String>;
    fn result_type(&self, index: usize) -> Option<Arc<RwLock<dyn Type>>>;
    fn results(&self) -> Values;
    fn set_anonymous_result(&self, result_type: Arc<RwLock<dyn Type>>) -> Result<()>;
    fn set_argument(&self, index: usize, argument: Arc<RwLock<Value>>);
    fn set_attributes(&self, attributes: Attributes);
    fn set_name(&self, name: OperationName);
    fn set_operand(&self, index: usize, operand: Arc<RwLock<OpOperand>>);
    fn set_operands(&self, operands: OpOperands);
    fn set_parent(&self, parent: Option<Arc<RwLock<Block>>>);
    fn set_region(&self, region: Option<Arc<RwLock<Region>>>);
    fn set_results(&self, results: Values);
    fn successors(&self) -> Vec<Arc<RwLock<dyn Op>>>;
}

impl GuardedOperation for Arc<RwLock<Operation>> {
    fn arguments(&self) -> Values {
        self.try_read().unwrap().arguments()
    }
    fn attributes(&self) -> Attributes {
        self.try_read().unwrap().attributes()
    }
    fn blocks(&self) -> Vec<Arc<RwLock<Block>>> {
        self.try_read().unwrap().blocks()
    }
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        self.try_read().unwrap().display(f, indent)
    }
    fn name(&self) -> OperationName {
        self.try_read().unwrap().name()
    }
    fn operand(&self, index: usize) -> Option<Arc<RwLock<OpOperand>>> {
        self.try_read().unwrap().operand(index)
    }
    fn operands(&self) -> OpOperands {
        self.try_read().unwrap().operands()
    }
    fn parent(&self) -> Option<Arc<RwLock<Block>>> {
        let operation = self.try_read().unwrap();
        operation.parent()
    }
    fn predecessors(&self) -> Vec<Arc<RwLock<dyn Op>>> {
        self.try_read().unwrap().predecessors()
    }
    fn region(&self) -> Option<Arc<RwLock<Region>>> {
        self.try_read().unwrap().region()
    }
    fn rename_variables(&self, renamer: &dyn VariableRenamer) -> Result<()> {
        self.try_read().unwrap().rename_variables(renamer)
    }
    fn result(&self, index: usize) -> Option<Arc<RwLock<Value>>> {
        self.try_read().unwrap().result(index)
    }
    fn result_names(&self) -> Vec<String> {
        self.try_read().unwrap().result_names()
    }
    fn result_type(&self, index: usize) -> Option<Arc<RwLock<dyn Type>>> {
        self.try_read().unwrap().result_type(index)
    }
    fn results(&self) -> Values {
        self.try_read().unwrap().results()
    }
    fn set_anonymous_result(&self, result_type: Arc<RwLock<dyn Type>>) -> Result<()> {
        self.try_write().unwrap().set_anonymous_result(result_type)
    }
    fn set_argument(&self, index: usize, argument: Arc<RwLock<Value>>) {
        self.try_write().unwrap().set_argument(index, argument);
    }
    fn set_attributes(&self, attributes: Attributes) {
        self.try_write().unwrap().set_attributes(attributes);
    }
    fn set_name(&self, name: OperationName) {
        self.try_write().unwrap().set_name(name);
    }
    fn set_operand(&self, index: usize, operand: Arc<RwLock<OpOperand>>) {
        self.try_write().unwrap().set_operand(index, operand);
    }
    fn set_operands(&self, operands: OpOperands) {
        self.try_write().unwrap().set_operands(operands);
    }
    fn set_region(&self, region: Option<Arc<RwLock<Region>>>) {
        self.try_write().unwrap().set_region(region);
    }
    fn set_parent(&self, parent: Option<Arc<RwLock<Block>>>) {
        self.try_write().unwrap().set_parent(parent);
    }
    fn set_results(&self, results: Values) {
        self.try_write().unwrap().set_results(results);
    }
    fn successors(&self) -> Vec<Arc<RwLock<dyn Op>>> {
        self.try_read().unwrap().successors()
    }
}
