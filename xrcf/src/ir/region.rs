use crate::ir::block::Block;
use crate::ir::Op;
use crate::ir::UnsetBlock;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::RwLock;

/// A list of blocks.
pub struct Region {
    blocks: Vec<Arc<RwLock<Block>>>,
    /// This field does not have to be an `Arc<RwLock<..>>` because
    /// the `Region` is shared via `Arc<RwLock<..>>`.
    parent: Option<Arc<RwLock<dyn Op>>>,
}

impl Region {
    pub fn new(blocks: Vec<Arc<RwLock<Block>>>, parent: Option<Arc<RwLock<dyn Op>>>) -> Self {
        Self { blocks, parent }
    }
    pub fn blocks(&self) -> Vec<Arc<RwLock<Block>>> {
        self.blocks.clone()
    }
    pub fn block(&self, index: usize) -> Arc<RwLock<Block>> {
        self.blocks[index].clone()
    }
    pub fn parent(&self) -> Option<Arc<RwLock<dyn Op>>> {
        self.parent.clone()
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
    pub fn add_new_block(&mut self) -> UnsetBlock {
        let block = Block::default();
        let block = Arc::new(RwLock::new(block));
        self.blocks_mut().push(block.clone());
        UnsetBlock::new(block)
    }
    pub fn set_parent(&mut self, parent: Option<Arc<RwLock<dyn Op>>>) {
        self.parent = parent;
    }
    pub fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        write!(f, " {{\n")?;
        for block in self.blocks() {
            let block = block.read().unwrap();
            block.display(f, indent + 1)?;
        }
        let spaces = crate::ir::spaces(indent);
        write!(f, "{spaces}}}")
    }
}

impl Display for Region {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}

impl Default for Region {
    fn default() -> Self {
        Self {
            blocks: vec![],
            parent: None,
        }
    }
}

pub trait GuardedRegion {
    fn blocks(&self) -> Vec<Arc<RwLock<Block>>>;
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result;
    fn set_blocks(&self, blocks: Vec<Arc<RwLock<Block>>>);
    fn set_parent(&self, parent: Option<Arc<RwLock<dyn Op>>>);
}

impl GuardedRegion for Arc<RwLock<Region>> {
    fn blocks(&self) -> Vec<Arc<RwLock<Block>>> {
        self.try_read().unwrap().blocks()
    }
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        self.try_read().unwrap().display(f, indent)
    }
    fn set_blocks(&self, blocks: Vec<Arc<RwLock<Block>>>) {
        self.try_write().unwrap().set_blocks(blocks);
    }
    fn set_parent(&self, parent: Option<Arc<RwLock<dyn Op>>>) {
        self.try_write().unwrap().set_parent(parent);
    }
}
