use crate::ir::Op;
use std::fmt::Debug;
use std::fmt::Display;
use std::sync::Arc;
use std::sync::RwLock;

#[derive(Clone, Debug)]
pub struct Type {
    name: String,
}

impl Type {
    pub fn new(name: String) -> Self {
        Type { name }
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

#[derive(Debug)]
pub struct BlockArgument {
    name: String,
    typ: Type,
}

impl BlockArgument {
    pub fn new(name: String, typ: Type) -> Self {
        BlockArgument { name, typ }
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn set_name(&mut self, name: &str) {
        self.name = name.to_string();
    }
    pub fn typ(&self) -> &Type {
        &self.typ
    }
}

impl Display for BlockArgument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} : {}", self.name, self.typ)
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
    /// Return the type via the defining op.
    pub fn typ(&self) -> Type {
        let defining_op = self.defining_op();
        let defining_op = defining_op.try_read().unwrap();
        let operation = defining_op.operation();
        let operation = operation.try_read().unwrap();
        let result_types = operation.result_types();
        let result_types = result_types.try_read().unwrap();
        assert!(
            result_types.len() == 1,
            "Not implemented for multiple results"
        );
        let typ = result_types[0].read().unwrap().clone();
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
    pub fn name(&self) -> &str {
        match self {
            Value::BlockArgument(arg) => &arg.name,
            Value::OpResult(result) => &result.name,
        }
    }
    pub fn typ(&self) -> Type {
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
            Value::BlockArgument(arg) => arg.set_name(name),
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
            let operands = operation.operands();
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
            Value::BlockArgument(arg) => write!(f, "{} : {}", arg.name, arg.typ.name),
            Value::OpResult(result) => write!(f, "{}", result.name),
        }
    }
}

pub struct OpOperand {
    pub value: Arc<RwLock<Value>>,
}

impl OpOperand {
    pub fn new(value: Arc<RwLock<Value>>) -> Self {
        OpOperand { value }
    }
    pub fn name(&self) -> String {
        let value = self.value.try_read().unwrap();
        value.name().to_string()
    }
    pub fn value(&self) -> Arc<RwLock<Value>> {
        self.value.clone()
    }
    /// If this `OpOperand` is the result of an operation, return the operation
    /// that defines it.
    pub fn defining_op(&self) -> Option<Arc<RwLock<dyn Op>>> {
        let value = self.value();
        let value = &*value.read().unwrap();
        match value {
            Value::BlockArgument(_) => None,
            Value::OpResult(op_res) => Some(op_res.defining_op()),
        }
    }
}

impl Display for OpOperand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}