use crate::ir::block::Block;
use std::fmt::Display;
use std::sync::Arc;
use std::sync::RwLock;

/// A list of blocks.
pub struct Region {
    blocks: Vec<Arc<RwLock<Block>>>,
}

impl Region {
    pub fn new(blocks: Vec<Arc<RwLock<Block>>>) -> Self {
        Self { blocks }
    }
    pub fn blocks(&self) -> &Vec<Arc<RwLock<Block>>> {
        &self.blocks
    }
    pub fn blocks_mut(&mut self) -> &mut Vec<Arc<RwLock<Block>>> {
        &mut self.blocks
    }
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
    pub fn set_blocks(&mut self, blocks: Vec<Arc<RwLock<Block>>>) {
        self.blocks = blocks;
    }
}

impl Display for Region {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{\n")?;
        for block in self.blocks() {
            write!(f, "  {}", block.read().unwrap())?;
        }
        write!(f, "\n}}")
    }
}

impl Default for Region {
    fn default() -> Self {
        Self { blocks: vec![] }
    }
}
