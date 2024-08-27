use crate::ir::operation::Operation;

#[derive(Clone)]
pub struct BlockArgument {
    name: String,
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
}
