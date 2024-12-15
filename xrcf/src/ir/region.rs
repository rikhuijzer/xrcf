use crate::ir::block::Block;
use crate::ir::Blocks;
use crate::ir::GuardedBlock;
use crate::ir::Op;
use crate::ir::UnsetBlock;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::RwLock;

/// A list of blocks.
pub struct Region {
    /// Blocks in the region.
    ///
    /// This field is an `Arc<RwLock<...>>` because the parser may read the
    /// blocks before the region is fully constructed.
    blocks: Blocks,
    // This field does not have to be an `Arc<RwLock<..>>` because the `Region`
    // is shared via `Arc<RwLock<..>>`.
    parent: Option<Arc<RwLock<dyn Op>>>,
}

impl Region {
    pub fn new(blocks: Blocks, parent: Option<Arc<RwLock<dyn Op>>>) -> Self {
        Self { blocks, parent }
    }
    pub fn blocks(&self) -> Blocks {
        self.blocks.clone()
    }
    pub fn block(&self, index: usize) -> Arc<RwLock<Block>> {
        self.blocks.vec().try_read().unwrap()[index].clone()
    }
    pub fn parent(&self) -> Option<Arc<RwLock<dyn Op>>> {
        self.parent.clone()
    }
    pub fn ops(&self) -> Vec<Arc<RwLock<dyn Op>>> {
        let mut result = Vec::new();
        let blocks = self.blocks();
        let blocks = blocks.vec();
        let blocks = blocks.try_read().unwrap();
        for block in blocks.iter() {
            let ops = block.ops();
            let ops = ops.read().unwrap();
            for op in ops.iter() {
                result.push(op.clone());
            }
        }
        result
    }
    /// Return the index of `block` in `self`.
    ///
    /// Returns `None` if `block` is not found in `self`.
    pub fn index_of(&self, block: &Block) -> Option<usize> {
        self.blocks().index_of(block)
    }
    pub fn is_empty(&self) -> bool {
        let blocks = self.blocks();
        let blocks = blocks.vec();
        let blocks = blocks.try_read().unwrap();
        blocks.is_empty()
    }
    pub fn set_blocks(&mut self, blocks: Blocks) {
        self.blocks = blocks;
    }
    pub fn add_empty_block(&mut self) -> UnsetBlock {
        let block = Block::default();
        let block = Arc::new(RwLock::new(block));
        let blocks = self.blocks();
        let blocks = blocks.vec();
        let mut blocks = blocks.try_write().unwrap();
        blocks.push(block.clone());
        UnsetBlock::new(block)
    }
    pub fn set_parent(&mut self, parent: Option<Arc<RwLock<dyn Op>>>) {
        self.parent = parent;
    }
    pub fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        write!(f, " {{\n")?;
        let blocks = self.blocks();
        let blocks = blocks.vec();
        let blocks = blocks.try_read().unwrap();
        for block in blocks.iter() {
            let block = block.try_read().unwrap();
            block.display(f, indent + 1)?;
        }
        let spaces = crate::ir::spaces(indent);
        write!(f, "{spaces}}}")
    }
    /// Find a unique name for a block (for example, `bb2`).
    pub fn unique_block_name(&self) -> String {
        let blocks = self.blocks();
        let blocks = blocks.vec();
        let blocks = blocks.try_read().unwrap();
        let mut new_name: i32 = 0;
        for block in blocks.iter() {
            let block = block.try_read().unwrap();
            let label = block.label();
            if let Some(label) = label {
                let name = label.trim_start_matches("bb").trim_start_matches("^bb");
                if let Ok(num) = name.parse::<i32>() {
                    // Ensure the new name is greater than any existing name.
                    // This makes the output a bit clearer.
                    new_name = new_name.max(num);
                }
            }
        }
        new_name += 1;
        format!("bb{new_name}")
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
            blocks: Blocks::new(Arc::new(RwLock::new(vec![]))),
            parent: None,
        }
    }
}

pub trait GuardedRegion {
    fn add_empty_block(&self) -> UnsetBlock;
    fn blocks(&self) -> Blocks;
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result;
    fn ops(&self) -> Vec<Arc<RwLock<dyn Op>>>;
    fn set_blocks(&self, blocks: Blocks);
    fn set_parent(&self, parent: Option<Arc<RwLock<dyn Op>>>);
    fn unique_block_name(&self) -> String;
}

impl GuardedRegion for Arc<RwLock<Region>> {
    fn add_empty_block(&self) -> UnsetBlock {
        self.try_write().unwrap().add_empty_block()
    }
    fn blocks(&self) -> Blocks {
        self.try_read().unwrap().blocks()
    }
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        self.try_read().unwrap().display(f, indent)
    }
    fn ops(&self) -> Vec<Arc<RwLock<dyn Op>>> {
        self.try_read().unwrap().ops()
    }
    fn set_blocks(&self, blocks: Blocks) {
        self.try_write().unwrap().set_blocks(blocks);
    }
    fn set_parent(&self, parent: Option<Arc<RwLock<dyn Op>>>) {
        self.try_write().unwrap().set_parent(parent);
    }
    fn unique_block_name(&self) -> String {
        self.try_read().unwrap().unique_block_name()
    }
}
