use crate::ir::ModuleOp;
use crate::ir::Op;

pub fn canonicalize(module: &mut ModuleOp) {
    let region = module.region();
    let region = region.read().unwrap();
    if let Some(region) = region.as_ref() {
        for block in region.blocks() {
            let block = block.read().unwrap();
            println!("block:\n{}", block);
            for op in block.ops() {
                let op = op.as_ref();
                println!("op: {}", op);
            }
        }
    }
}
