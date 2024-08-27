use crate::ir::block::Block;
use std::fmt::Display;

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

impl Display for Region {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{\n")?;
        for block in self.blocks() {
            write!(f, "  {}", block)?;
        }
        write!(f, "\n}}")?;
        Ok(())
    }
}
