use crate::ir::Op;
use std::sync::Arc;

pub fn canonicalize(op: &mut dyn Op) {
    let region = op.region();
    let region = region.read().unwrap();
    if let Some(region) = region.as_ref() {
        for block in region.blocks() {
            let mut block = block.write().unwrap();
            println!("block:\n{}", block);
            let ops = Arc::make_mut(block.ops_mut());
            for op in ops.iter_mut() {
                if let Some(op) = Arc::get_mut(op) {
                    canonicalize(op);
                }
            }
        }
    }
}
