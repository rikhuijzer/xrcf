use crate::ir::Op;
use std::fmt::Display;
use std::sync::Arc;

#[derive(Clone)]
pub struct BlockArgument {
    name: String,
}

pub struct Block {
    label: String,
    arguments: Vec<BlockArgument>,
    ops: Vec<Arc<dyn Op>>,
}

impl Block {
    pub fn new(label: String, arguments: Vec<BlockArgument>, ops: Vec<Arc<dyn Op>>) -> Self {
        Self {
            label,
            arguments,
            ops,
        }
    }
    pub fn ops(&self) -> Vec<Arc<dyn Op>> {
        self.ops.clone()
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
