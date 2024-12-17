use crate::ir::block::Block;
use crate::ir::BlockName;
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
    blocks: Arc<RwLock<Vec<Arc<RwLock<Block>>>>>,
    // This field does not have to be an `Arc<RwLock<..>>` because the `Region`
    // is shared via `Arc<RwLock<..>>`.
    parent: Option<Arc<RwLock<dyn Op>>>,
}

/// Set fresh block labels for all blocks in the region.
///
/// This happens before the individual blocks are printed since some operands
/// will point to blocks that occur later. If we would refresh the name during
/// printing of the block (like we do with ssa variables), then the operands
/// would print outdated names.
fn set_fresh_block_labels(blocks: &Vec<Arc<RwLock<Block>>>) {
    let mut label_index: usize = 1;
    for block in blocks.iter() {
        let block = block.try_read().unwrap();
        let label = block.label();
        let label_read = label.try_read().unwrap();
        match &*label_read {
            BlockName::Name(_name) => {
                drop(label_read);
                let new = format!("^bb{label_index}");
                let new = BlockName::Name(new);
                block.set_label(new);
                label_index += 1;
            }
            BlockName::Unnamed => {}
            BlockName::Unset => {
                drop(label_read);
                let new = format!("^bb{label_index}");
                let new = BlockName::Name(new);
                block.set_label(new);
                label_index += 1;
            }
        }
    }
}

impl Region {
    pub fn new(
        blocks: Arc<RwLock<Vec<Arc<RwLock<Block>>>>>,
        parent: Option<Arc<RwLock<dyn Op>>>,
    ) -> Self {
        Self { blocks, parent }
    }
    pub fn blocks(&self) -> Arc<RwLock<Vec<Arc<RwLock<Block>>>>> {
        self.blocks.clone()
    }
    pub fn block(&self, index: usize) -> Arc<RwLock<Block>> {
        self.blocks.read().unwrap()[index].clone()
    }
    pub fn parent(&self) -> Option<Arc<RwLock<dyn Op>>> {
        self.parent.clone()
    }
    pub fn ops(&self) -> Vec<Arc<RwLock<dyn Op>>> {
        let mut result = Vec::new();
        let blocks = self.blocks();
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
    pub fn index_of(&self, block: &Block) -> Option<usize> {
        let blocks = self.blocks();
        let blocks = blocks.try_read().unwrap();
        if blocks.is_empty() {
            panic!("Region {self} has no blocks");
        }
        for (i, current) in (&blocks).iter().enumerate() {
            let current = current.try_read().unwrap();
            if *current == *block {
                return Some(i);
            }
        }
        None
    }
    pub fn is_empty(&self) -> bool {
        let blocks = self.blocks();
        let blocks = blocks.try_read().unwrap();
        blocks.is_empty()
    }
    pub fn set_blocks(&mut self, blocks: Arc<RwLock<Vec<Arc<RwLock<Block>>>>>) {
        self.blocks = blocks;
    }
    pub fn add_empty_block(&mut self) -> UnsetBlock {
        let block = Block::default();
        let block = Arc::new(RwLock::new(block));
        let blocks = self.blocks();
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
        let blocks = blocks.try_read().unwrap();
        set_fresh_block_labels(&blocks);
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
        let blocks = blocks.try_read().unwrap();
        let mut new_name: i32 = 0;
        for block in blocks.iter() {
            let block = block.try_read().unwrap();
            let label = block.label();
            let label = label.try_read().unwrap();
            match &*label {
                BlockName::Name(name) => {
                    let name = name.trim_start_matches("bb").trim_start_matches("^bb");
                    if let Ok(num) = name.parse::<i32>() {
                        // Ensure the new name is greater than any existing name.
                        // This makes the output a bit clearer.
                        new_name = new_name.max(num);
                    }
                }
                BlockName::Unset => {}
                BlockName::Unnamed => {}
            }
        }
        new_name += 1;
        format!("^bb{new_name}")
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
            blocks: Arc::new(RwLock::new(vec![])),
            parent: None,
        }
    }
}

pub trait GuardedRegion {
    fn add_empty_block(&self) -> UnsetBlock;
    fn blocks(&self) -> Arc<RwLock<Vec<Arc<RwLock<Block>>>>>;
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result;
    fn ops(&self) -> Vec<Arc<RwLock<dyn Op>>>;
    fn set_blocks(&self, blocks: Arc<RwLock<Vec<Arc<RwLock<Block>>>>>);
    fn set_parent(&self, parent: Option<Arc<RwLock<dyn Op>>>);
    fn unique_block_name(&self) -> String;
}

impl GuardedRegion for Arc<RwLock<Region>> {
    fn add_empty_block(&self) -> UnsetBlock {
        self.try_write().unwrap().add_empty_block()
    }
    fn blocks(&self) -> Arc<RwLock<Vec<Arc<RwLock<Block>>>>> {
        self.try_read().unwrap().blocks()
    }
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        self.try_read().unwrap().display(f, indent)
    }
    fn ops(&self) -> Vec<Arc<RwLock<dyn Op>>> {
        self.try_read().unwrap().ops()
    }
    fn set_blocks(&self, blocks: Arc<RwLock<Vec<Arc<RwLock<Block>>>>>) {
        self.try_write().unwrap().set_blocks(blocks);
    }
    fn set_parent(&self, parent: Option<Arc<RwLock<dyn Op>>>) {
        self.try_write().unwrap().set_parent(parent);
    }
    fn unique_block_name(&self) -> String {
        self.try_read().unwrap().unique_block_name()
    }
}
