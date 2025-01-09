use crate::ir::block::Block;
use crate::ir::BlockName;
use crate::ir::Blocks;
use crate::ir::Op;
use crate::ir::UnsetBlock;
use crate::ir::Value;
use crate::shared::Shared;
use crate::shared::SharedExt;
use std::fmt::Display;
use std::fmt::Formatter;

/// A list of blocks.
pub struct Region {
    /// Blocks in the region.
    ///
    /// This field is an `Arc<RwLock<...>>` because the parser may read the
    /// blocks before the region is fully constructed.
    blocks: Blocks,
    // This field does not have to be an `Arc<RwLock<..>>` because the `Region`
    // is shared via `Arc<RwLock<..>>`.
    parent: Option<Shared<dyn Op>>,
}

/// Set fresh block arguments for all blocks in the region.
///
/// This happens during printing because the names are not used during the
/// rewrite phase.
fn set_fresh_block_argument_names(prefix: &str, blocks: &[Shared<Block>]) {
    // Each block argument in a region needs an unique name because
    // arguments stay in scope until the end of the region.
    let mut name_index: usize = 0;
    for block in blocks.iter() {
        let block = block.rd();
        for argument in block.arguments() {
            let name = format!("{prefix}{name_index}");
            argument.wr().set_name(&name);
            name_index += 1;
        }
    }
}

/// Set fresh block labels for all blocks in the region.
///
/// This happens before the individual blocks are printed since some operands
/// will point to blocks that are defined later. If we would refresh the name
/// during printing of the block, then the operands would print outdated names.
fn set_fresh_block_labels(prefix: &str, blocks: &[Shared<Block>]) {
    let mut label_index: usize = 1;
    for block in blocks.iter() {
        let block = block.rd();
        let label = block.label();
        let label_read = label.rd();
        match &*label_read {
            BlockName::Name(_name) => {
                drop(label_read);
                let new = format!("{prefix}{label_index}");
                let new = BlockName::Name(new);
                block.set_label(new);
                label_index += 1;
            }
            BlockName::Unnamed => {}
            BlockName::Unset => {
                drop(label_read);
                let new = format!("{prefix}{label_index}");
                let new = BlockName::Name(new);
                block.set_label(new);
                label_index += 1;
            }
        }
    }
}

/// Set fresh SSA names for all blocks in the region.
///
/// This happens during printing because the names are not used during the
/// rewrite phase.
fn set_fresh_ssa_names(prefix: &str, blocks: &[Shared<Block>]) {
    // SSA names stay in scope until the end of the region I think.
    let mut name_index: usize = 0;
    for block in blocks.iter() {
        for op in block.rd().ops().rd().iter() {
            for result in op.rd().operation().rd().results() {
                if let Value::OpResult(op_result) = &*result.rd() {
                    let name = format!("{prefix}{name_index}");
                    op_result.set_name(&name);
                    name_index += 1;
                }
            }
        }
    }
}

impl Region {
    pub fn new(blocks: Blocks, parent: Option<Shared<dyn Op>>) -> Self {
        Self { blocks, parent }
    }
    pub fn blocks(&self) -> Blocks {
        self.blocks.clone()
    }
    pub fn block(&self, index: usize) -> Shared<Block> {
        self.blocks().into_iter().nth(index).unwrap()
    }
    pub fn parent(&self) -> Option<Shared<dyn Op>> {
        self.parent.clone()
    }
    pub fn ops(&self) -> Vec<Shared<dyn Op>> {
        let mut result = Vec::new();
        for block in self.blocks().into_iter() {
            for op in block.rd().ops().rd().iter() {
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
    pub fn add_empty_block_before(&self, block: Shared<Block>) -> UnsetBlock {
        let index = self.index_of(&block.rd()).unwrap();

        let new = Block::default();
        let new = Shared::new(new.into());
        self.blocks().vec().wr().insert(index, new.clone());
        UnsetBlock::new(new)
    }
    pub fn set_parent(&mut self, parent: Option<Shared<dyn Op>>) {
        self.parent = parent;
    }
    pub fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        writeln!(f, " {{")?;
        let blocks = self.blocks();
        let blocks = blocks.vec();
        let blocks = blocks.rd();
        // The parent could be a module which is not always rewritten. Ops below
        // the module have to be rewritten for the output to be correct so will
        // know the right prefixes.
        let prefixes = self
            .block(0)
            .rd()
            .ops()
            .rd()
            .first()
            .unwrap()
            .rd()
            .prefixes();
        set_fresh_block_argument_names(prefixes.argument, &blocks);
        set_fresh_block_labels(prefixes.block, &blocks);
        set_fresh_ssa_names(prefixes.ssa, &blocks);
        for block in blocks.iter() {
            block.rd().display(f, indent + 1)?;
        }
        write!(f, "{}}}", crate::ir::spaces(indent))
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
