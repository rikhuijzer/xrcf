use crate::ir::block::Block;
use crate::ir::BlockName;
use crate::ir::Blocks;
use crate::ir::GuardedBlock;
use crate::ir::Op;
use crate::ir::UnsetBlock;
use crate::shared::Shared;
use crate::shared::SharedExt;
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

/// Set fresh block labels for all blocks in the region.
///
/// This happens before the individual blocks are printed since some operands
/// will point to blocks that occur later. If we would refresh the name during
/// printing of the block (like we do with ssa variables), then the operands
/// would print outdated names.
fn set_fresh_block_labels(blocks: &Vec<Arc<RwLock<Block>>>) {
    let mut label_index: usize = 1;
    for block in blocks.iter() {
        let block = block.rd();
        let label_prefix = block.label_prefix();
        let label = block.label();
        let label_read = label.rd();
        match &*label_read {
            BlockName::Name(_name) => {
                drop(label_read);
                let new = format!("{label_prefix}bb{label_index}");
                let new = BlockName::Name(new);
                block.set_label(new);
                label_index += 1;
            }
            BlockName::Unnamed => {}
            BlockName::Unset => {
                drop(label_read);
                let new = format!("{label_prefix}bb{label_index}");
                let new = BlockName::Name(new);
                block.set_label(new);
                label_index += 1;
            }
        }
    }
}

impl Region {
    pub fn new(blocks: Blocks, parent: Option<Arc<RwLock<dyn Op>>>) -> Self {
        Self { blocks, parent }
    }
    pub fn blocks(&self) -> Blocks {
        self.blocks.clone()
    }
    pub fn block(&self, index: usize) -> Arc<RwLock<Block>> {
        self.blocks().into_iter().nth(index).unwrap()
    }
    pub fn parent(&self) -> Option<Arc<RwLock<dyn Op>>> {
        self.parent.clone()
    }
    pub fn ops(&self) -> Vec<Arc<RwLock<dyn Op>>> {
        let mut result = Vec::new();
        for block in self.blocks().into_iter() {
            for op in block.ops().rd().iter() {
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
    pub fn set_blocks(&mut self, blocks: Blocks) {
        self.blocks = blocks;
    }
    pub fn add_empty_block(&self) -> UnsetBlock {
        let block = Block::default();
        let block = Shared::new(block.into());
        self.blocks().vec().wr().push(block.clone());
        UnsetBlock::new(block)
    }
    pub fn add_empty_block_before(&self, block: Arc<RwLock<Block>>) -> UnsetBlock {
        let index = self.index_of(&block.rd()).unwrap();

        let new = Block::default();
        let new = Shared::new(new.into());
        self.blocks().vec().wr().insert(index, new.clone());
        UnsetBlock::new(new)
    }
    pub fn set_parent(&mut self, parent: Option<Arc<RwLock<dyn Op>>>) {
        self.parent = parent;
    }
    pub fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        write!(f, " {{\n")?;
        let blocks = self.blocks();
        let blocks = blocks.vec();
        let blocks = blocks.rd();
        set_fresh_block_labels(&blocks);
        for block in blocks.iter() {
            block.rd().display(f, indent + 1)?;
        }
        write!(f, "{}}}", crate::ir::spaces(indent))
    }
    /// Find a unique name for a block (for example, `bb2`).
    pub fn unique_block_name(&self) -> String {
        let mut new_name: i32 = 0;
        for block in self.blocks().into_iter() {
            match &*block.rd().label().rd() {
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
            blocks: Blocks::new(Shared::new(vec![].into())),
            parent: None,
        }
    }
}
