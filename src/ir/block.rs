use crate::ir::operation::Operation;
use std::fmt::Display;

#[derive(Clone)]
pub struct BlockArgument {
    name: String,
}

impl Display for BlockArgument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

#[derive(Clone)]
pub struct Block {
    label: String,
    arguments: Vec<BlockArgument>,
    operations: Vec<Operation>,
}

impl Block {
    pub fn new(label: String, arguments: Vec<BlockArgument>, operations: Vec<Operation>) -> Self {
        Self { label, arguments, operations }
    }
    pub fn operations(&self) -> Vec<Operation> {
        self.operations.to_vec()
    }
}

impl Display for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label)?;
        for operation in self.operations() {
            write!(f, " {}", operation)?;
        }
        Ok(())
    }
}
