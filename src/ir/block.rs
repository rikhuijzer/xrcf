use std::fmt::Display;
use crate::ir::Op;

#[derive(Clone)]
pub struct BlockArgument {
    name: String,
}

pub struct Block {
    label: String,
    arguments: Vec<BlockArgument>,
    ops: Vec<Box<dyn Op>>,
}

impl Block {
    pub fn new(label: String, arguments: Vec<BlockArgument>, ops: Vec<Box<dyn Op>>) -> Self {
        Self {
            label,
            arguments,
            ops,
        }
    }
    pub fn ops(&self) -> Vec<Box<dyn Op>> {
        todo!()
    }
}

impl Display for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if !self.label.is_empty() {
            write!(f, "{} ", self.label)?;
        }
        for op in self.ops() {
            write!(f, "{} ", op)?;
        }
        Ok(())
    }
}
