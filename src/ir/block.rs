use crate::ir::Op;
use crate::ir::Region;
use std::fmt::Display;
use std::sync::Arc;
use std::sync::RwLock;

#[derive(Clone)]
pub struct BlockArgument {
    name: String,
}

pub struct Block {
    label: Option<String>,
    arguments: Vec<BlockArgument>,
    ops: Vec<Arc<dyn Op>>,
    parent: Arc<RwLock<Region>>,
}

impl Block {
    pub fn new(
        label: Option<String>,
        arguments: Vec<BlockArgument>,
        ops: Vec<Arc<dyn Op>>,
        parent: Arc<RwLock<Region>>,
    ) -> Self {
        Self {
            label,
            arguments,
            ops,
            parent,
        }
    }
    pub fn ops(&self) -> Vec<Arc<dyn Op>> {
        self.ops.clone()
    }
    pub fn parent(&self) -> Arc<RwLock<Region>> {
        self.parent.clone()
    }
    pub fn display(&self, f: &mut std::fmt::Formatter<'_>, indent: i32) -> std::fmt::Result {
        if let Some(label) = &self.label {
            write!(f, "{} ", label)?;
        }
        for op in self.ops() {
            op.display(f, indent)?;
        }
        Ok(())
    }
}

impl Display for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let indent = 0;
        self.display(f, indent)
    }
}
