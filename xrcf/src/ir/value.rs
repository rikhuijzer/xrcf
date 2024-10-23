use crate::ir::Op;
use crate::ir::OpOperand;
use crate::ir::Operation;
use crate::ir::Type;
use crate::parser::Parser;
use crate::parser::ParserDispatch;
use crate::parser::TokenKind;
use anyhow::Result;
use std::fmt::Display;
use std::sync::Arc;
use std::sync::RwLock;

pub struct BlockArgument {
    /// The name of the block argument. Does not have to be set because
    /// anonymous arguments are allowed for functions without an implementation.
    name: Option<String>,
    typ: Arc<RwLock<dyn Type>>,
}

impl BlockArgument {
    pub fn new(name: Option<String>, typ: Arc<RwLock<dyn Type>>) -> Self {
        BlockArgument { name, typ }
    }
    pub fn name(&self) -> Option<String> {
        self.name.clone()
    }
    pub fn set_name(&mut self, name: Option<String>) {
        self.name = name;
    }
    pub fn typ(&self) -> Arc<RwLock<dyn Type>> {
        self.typ.clone()
    }
}

impl Display for BlockArgument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let typ = self.typ.try_read().unwrap();
        match &self.name {
            Some(name) => write!(f, "{} : {}", name, typ),
            None => write!(f, "{}", typ),
        }
    }
}

pub struct OpResult {
    name: String,
    defining_op: Option<Arc<RwLock<dyn Op>>>,
}

impl OpResult {
    pub fn new(name: String, defining_op: Option<Arc<RwLock<dyn Op>>>) -> Self {
        OpResult { name, defining_op }
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Return the type of the op result via the defining op.
    pub fn typ(&self) -> Arc<RwLock<dyn Type>> {
        let defining_op = self.defining_op();
        let defining_op = defining_op.try_read().unwrap();
        let operation = defining_op.operation();
        let operation = operation.try_read().unwrap();
        let result_types = operation.result_types();
        let result_types = result_types.vec();
        let result_types = result_types.try_read().unwrap();
        assert!(
            result_types.len() == 1,
            "Expected one result type for {} from {}, but got {}",
            self.name,
            operation.name(),
            result_types.len(),
        );
        let typ = result_types[0].clone();
        typ
    }
    pub fn defining_op(&self) -> Arc<RwLock<dyn Op>> {
        if self.defining_op.is_none() {
            panic!("Defining op not set for {}", self.name);
        }
        self.defining_op.clone().unwrap()
    }
    pub fn set_name(&mut self, name: &str) {
        self.name = name.to_string();
    }
    pub fn set_defining_op(&mut self, op: Option<Arc<RwLock<dyn Op>>>) {
        self.defining_op = op;
    }
}

impl Default for OpResult {
    fn default() -> Self {
        Self {
            name: "<unset name>".to_string(),
            defining_op: None,
        }
    }
}

impl Display for OpResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

pub enum Users {
    /// The operation defines no `OpResult`s.
    HasNoOpResults,
    /// The operation defines `OpResult`s (and can still have zero users).
    OpOperands(Vec<Arc<RwLock<OpOperand>>>),
}

impl Users {
    pub fn len(&self) -> usize {
        match self {
            Users::HasNoOpResults => 0,
            Users::OpOperands(users) => users.len(),
        }
    }
}

/// Represents an instance of an SSA value in the IR,
/// representing a computable value that has a type and a set of users. An SSA
/// value is either a BlockArgument or the result of an operation. Note: This
/// class has value-type semantics and is just a simple wrapper around a
/// ValueImpl that is either owner by a block(in the case of a BlockArgument) or
/// an Operation(in the case of an OpResult).
/// As most IR construct, this isn't const-correct, but we keep method
/// consistent and as such methods that immediately modify this Value aren't
/// marked `const` (include modifying the Value use-list).
pub enum Value {
    BlockArgument(BlockArgument),
    OpResult(OpResult),
}

impl Value {
    /// The name of the value.
    ///
    /// Returns `None` for block arguments that do not have a name.
    pub fn name(&self) -> Option<String> {
        match self {
            Value::BlockArgument(arg) => arg.name.clone(),
            Value::OpResult(result) => Some(result.name.clone()),
        }
    }
    pub fn typ(&self) -> Arc<RwLock<dyn Type>> {
        match self {
            Value::BlockArgument(arg) => arg.typ.clone(),
            Value::OpResult(result) => result.typ(),
        }
    }
    pub fn set_defining_op(&mut self, op: Option<Arc<RwLock<dyn Op>>>) {
        match self {
            Value::BlockArgument(_) => panic!("Cannot set defining op for BlockArgument"),
            Value::OpResult(op_res) => op_res.set_defining_op(op),
        }
    }
    pub fn set_name(&mut self, name: &str) {
        match self {
            Value::BlockArgument(arg) => arg.set_name(Some(name.to_string())),
            Value::OpResult(result) => result.set_name(name),
        }
    }
    fn op_result_users(&self, op_res: &OpResult) -> Vec<Arc<RwLock<OpOperand>>> {
        let op = op_res.defining_op();
        let op = op.try_read().unwrap();
        let parent = {
            let operation = op.operation();
            let operation = operation.try_read().unwrap();
            let parent = operation.parent();
            parent.unwrap()
        };
        let block = parent.try_read().unwrap();
        let index = block.index_of(op.operation().clone());
        let index = index.unwrap();
        let ops = block.ops();
        let ops = ops.try_read().unwrap();
        let mut out = Vec::new();
        for i in index..ops.len() {
            let op = ops[i].try_read().unwrap();
            let operation = op.operation();
            let operation = operation.try_read().unwrap();
            let operands = operation.operands().vec();
            let operands = operands.try_read().unwrap();
            for operand in operands.iter() {
                let operand_clone = operand.clone();
                let operand_clone = operand_clone.try_read().unwrap();
                let value = operand_clone.value();
                let value = value.try_read().unwrap();
                if std::ptr::eq(&*value as *const Value, self as *const Value) {
                    out.push(operand.clone());
                }
            }
        }
        out
    }
    pub fn users(&self) -> Users {
        match self {
            Value::BlockArgument(_) => Users::HasNoOpResults,
            Value::OpResult(op_res) => Users::OpOperands(self.op_result_users(op_res)),
        }
    }
    /// Rename the value, and all its users.
    pub fn rename(&mut self, new_name: &str) {
        self.set_name(new_name);
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::BlockArgument(arg) => write!(f, "{arg}"),
            Value::OpResult(result) => write!(f, "{result}"),
        }
    }
}

/// Vector of values.
///
/// Used to store operation results and function arguments. This naming is
/// identical to MLIR values. Also there, a value can be a block argument or an
/// operation result.
#[derive(Clone)]
pub struct Values {
    values: Arc<RwLock<Vec<Arc<RwLock<Value>>>>>,
}

impl Values {
    pub fn from_vec(values: Vec<Arc<RwLock<Value>>>) -> Self {
        Values {
            values: Arc::new(RwLock::new(values)),
        }
    }
    pub fn vec(&self) -> Arc<RwLock<Vec<Arc<RwLock<Value>>>>> {
        self.values.clone()
    }
    /// Set the defining op for op results.
    ///
    /// This can be used when replacing an operation with another operation to
    /// set the defining op for the results to point to the new operation.
    ///
    /// Calling this on block arguments (like function arguments) will panic since
    /// block arguments do not specify a defining op.
    pub fn set_defining_op(&self, op: Arc<RwLock<dyn Op>>) {
        let results = self.values.read().unwrap();
        for result in results.iter() {
            let mut mut_result = result.try_write().unwrap();
            match &mut *mut_result {
                Value::BlockArgument(_) => {
                    panic!("This case should not occur")
                }
                Value::OpResult(res) => res.set_defining_op(Some(op.clone())),
            }
        }
    }
}

impl Default for Values {
    fn default() -> Self {
        Values {
            values: Arc::new(RwLock::new(vec![])),
        }
    }
}

impl Display for Values {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let joined = self
            .values
            .try_read()
            .unwrap()
            .iter()
            .map(|o| o.try_read().unwrap().to_string())
            .collect::<Vec<String>>()
            .join(", ");
        write!(f, "{joined}")
    }
}

// Putting these on the parser to allow method discovery via `parser.parse_`.
impl<T: ParserDispatch> Parser<T> {
    /// Parse `%arg0 : i64,`, or `i64,`.
    pub fn parse_function_argument(&mut self) -> Result<Arc<RwLock<Value>>> {
        if self.check(TokenKind::PercentIdentifier) {
            let identifier = self.expect(TokenKind::PercentIdentifier)?;
            let name = identifier.lexeme.clone();
            let _colon = self.expect(TokenKind::Colon)?;
            let typ = T::parse_type(self)?;
            let arg = Value::BlockArgument(BlockArgument::new(Some(name), typ));
            let operand = Arc::new(RwLock::new(arg));
            if self.check(TokenKind::Comma) {
                self.advance();
            }
            return Ok(operand);
        }
        if self.check(TokenKind::IntType) || self.check(TokenKind::Exclamation) {
            let typ = T::parse_type(self)?;
            let name = None;
            let arg = Value::BlockArgument(BlockArgument::new(name, typ));
            let operand = Arc::new(RwLock::new(arg));
            if self.check(TokenKind::Comma) {
                self.advance();
            }
            return Ok(operand);
        }
        Err(anyhow::anyhow!("Expected function argument"))
    }
    pub fn is_function_argument(&mut self) -> bool {
        // For example, `@add(%arg0 : i32)`.
        let perc = self.check(TokenKind::PercentIdentifier);
        // For example, `@add(i32)`.
        let int = self.check(TokenKind::IntType);
        // For example, `@printf(!llvm.ptr)`.
        let excl = self.check(TokenKind::Exclamation);
        perc || int || excl
    }
    /// Parse %0, %1.
    fn parse_op_results(&mut self) -> Result<Values> {
        let mut results = vec![];
        while self.check(TokenKind::PercentIdentifier) {
            let identifier = self.expect(TokenKind::PercentIdentifier)?;
            let name = identifier.lexeme.clone();
            let mut op_result = OpResult::default();
            op_result.set_name(&name);
            let result = Value::OpResult(op_result);
            results.push(Arc::new(RwLock::new(result)));
            if self.check(TokenKind::Equal) {
                let _equal = self.advance();
            }
        }
        let results = Arc::new(RwLock::new(results));
        let values = Values { values: results };
        Ok(values)
    }
    pub fn parse_op_results_into(&mut self, operation: &mut Operation) -> Result<Values> {
        let results = self.parse_op_results()?;
        operation.set_results(results.clone());
        Ok(results)
    }
    /// Parse `(%arg0 : i64, %arg1 : i64)`, or `(i64, !llvm.ptr)`.
    pub fn parse_function_arguments(&mut self) -> Result<Values> {
        let _lparen = self.expect(TokenKind::LParen)?;
        let mut operands = vec![];
        while self.is_function_argument() {
            operands.push(self.parse_function_argument()?);
        }
        let _rparen = self.expect(TokenKind::RParen);
        let values = Values::from_vec(operands);
        Ok(values)
    }
}
