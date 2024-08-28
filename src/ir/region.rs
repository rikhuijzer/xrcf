use crate::ir::block::Block;
use std::fmt::Display;
use std::pin::Pin;

/// A list of blocks.
pub struct Region {
    blocks: Vec<Pin<Box<Block>>>,
}

impl Region {
    pub fn new(blocks: Vec<Pin<Box<Block>>>) -> Self {
        Self { blocks }
    }
    pub fn blocks(&self) -> Vec<&Pin<Box<Block>>> {
        self.blocks.iter().collect()
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
