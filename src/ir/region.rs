use crate::ir::block::Block;

/// A list of blocks.
#[derive(Clone)]
pub struct Region {
    blocks: Vec<Block>,
}

impl Region {
    pub fn new(blocks: Vec<Block>) -> Self {
        Self { blocks }
    }
    pub fn blocks(&self) -> Vec<Block> {
        self.blocks.to_vec()
    }
    fn print(&self) -> String {
        todo!()
    }
}