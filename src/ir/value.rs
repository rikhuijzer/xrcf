use std::fmt::Display;

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

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::BlockArgument(arg) => write!(f, "{} : {}", arg.name, arg.typ.name),
            Value::OpResult(result) => write!(f, "{}", result.name),
        }
    }
}

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

pub struct BlockArgument {
    name: String,
    typ: Type,
}

impl BlockArgument {
    pub fn new(name: String, typ: Type) -> Self {
        BlockArgument { name, typ }
    }
}

pub struct OpResult {
    name: String,
    typ: Type,
}
