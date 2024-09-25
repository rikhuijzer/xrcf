use crate::ir::ModuleOp;
use crate::ir::Op;
use std::sync::Arc;

#[derive(Debug, PartialEq, Eq)]
pub enum CanonicalizeResult {
    Changed,
    Unchanged,
}

pub fn canonicalize_op(op: &mut dyn Op) -> CanonicalizeResult {
    let region = op.region();
    let region = region.read().unwrap();
    if let Some(region) = region.as_ref() {
        let mut changed = CanonicalizeResult::Unchanged;
        for block in region.blocks() {
            let mut block = block.write().unwrap();
            println!("block:\n{}", block);
            let ops = Arc::make_mut(block.ops_mut());
            for op in ops.iter_mut() {
                if let Some(op) = Arc::get_mut(op) {
                    let result = canonicalize_op(op);
                    if result == CanonicalizeResult::Changed {
                        changed = CanonicalizeResult::Changed;
                    }
                }
            }
        }
        changed
    } else {
        op.canonicalize()
    }
}

pub fn canonicalize(module: &mut ModuleOp) {
    let max_iterations = 10;
    for _ in 0..max_iterations {
        let result = canonicalize_op(module);
        if result == CanonicalizeResult::Unchanged {
            break;
        }
    }
}
