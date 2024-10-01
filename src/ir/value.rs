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
}

impl Display for BlockArgument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} : {}", self.name, self.typ)
    }
}

#[derive(Debug)]
pub struct OpResult {
    name: String,
    typ: Type,
}

impl OpResult {
    pub fn new(name: String, typ: Type) -> Self {
        OpResult { name, typ }
    }
    pub fn name(&self) -> &str {
        &self.name
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
#[derive(Debug)]
pub enum Value {
    BlockArgument(BlockArgument),
    OpResult(OpResult),
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
    pub operand_name: String,
}

impl OpOperand {
    pub fn new(value: Arc<RwLock<Value>>, operand_name: String) -> Self {
        OpOperand {
            value,
            operand_name,
        }
    }
    pub fn value(&self) -> Arc<RwLock<Value>> {
        self.value.clone()
    }
    pub fn operand_name(&self) -> &str {
        &self.operand_name
    }
}

impl Display for OpOperand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.operand_name)
    }
}
