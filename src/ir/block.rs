use crate::ir::operation::Operation;

struct BlockArgument {
    name: String,
}

pub struct Block {
    arguments: Vec<BlockArgument>,
    // operations: Vec<Box<dyn Operation>>,
}