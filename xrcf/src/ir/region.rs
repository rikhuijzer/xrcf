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

use super::BlockArgumentName;

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

/// Set fresh arguments names for all blocks in the region.
///
/// This happens during printing because the names are not used during the
/// rewrite phase.
fn set_fresh_argument_names(prefix: &str, blocks: &[Shared<Block>]) {
    // Each block argument in a region needs an unique name because
    // arguments stay in scope until the end of the region.
    let mut name_index: usize = 0;

    // The variables available in the region can also be the arguments of the
    // parent op (only if the parent op is a function). I think MLIR models this
    // by putting the arguments of the function is the first block. xrcf should
    // probably do the same.
    let parent_region = blocks.first().unwrap().rd().parent();
    let parent_region = parent_region.expect("parent not set");
    for op in parent_region.rd().ops().iter() {
        if op.rd().is_func() {
            for arg in op.rd().operation().rd().arguments().into_iter() {
                let mut arg = arg.wr();
                #[allow(clippy::single_match)]
                match &mut *arg {
                    Value::BlockArgument(block_arg) => match &mut block_arg.name {
                        BlockArgumentName::Name(_name) => {
                            let name = format!("{prefix}{name_index}");
                            block_arg.name = BlockArgumentName::Name(name);
                            name_index += 1;
                        }
                        BlockArgumentName::Unset => {
                            let name = format!("{prefix}{name_index}");
                            block_arg.name = BlockArgumentName::Name(name);
                            name_index += 1;
                        }
                        BlockArgumentName::Anonymous => {}
                    },
                    _ => {}
                }
            }
        }
    }

    for block in blocks.iter() {
        let block = block.rd();
        for arg in block.arguments() {
            arg.wr().set_name(&format!("{prefix}{name_index}"));
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
        let mut block = block.wr();
        match &block.label {
            BlockName::Name(_name) => {
                let new = format!("{prefix}{label_index}");
                let new = BlockName::Name(new);
                block.label = new;
                label_index += 1;
            }
            BlockName::Unnamed => {}
            BlockName::Unset => {
                let new = format!("{prefix}{label_index}");
                let new = BlockName::Name(new);
                block.label = new;
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
                if let Value::OpResult(op_result) = &mut *result.wr() {
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
    pub fn refresh_names(&self) {
        let blocks = self.blocks();
        let blocks = blocks.vec();
        let blocks = blocks.rd();
        // The parent could be a module which is not always rewritten. Ops below
        // the module have to be rewritten for the output to be correct so will
        // know the right prefixes, see also the [Op::prefixes] docstring for
        // more information.
        let prefixes = self
            .block(0)
            .rd()
            .ops()
            .rd()
            .first()
            .unwrap()
            .rd()
            .prefixes();
        set_fresh_argument_names(prefixes.argument, &blocks);
        set_fresh_block_labels(prefixes.block, &blocks);
        set_fresh_ssa_names(prefixes.ssa, &blocks);
    }
    pub fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        self.refresh_names();
        for block in self.blocks().vec().rd().iter() {
            block.rd().display(f, indent + 1)?;
        }
        Ok(())
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
