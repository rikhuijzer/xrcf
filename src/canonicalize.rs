use crate::ir::ModuleOp;
use crate::ir::Op;

#[derive(Debug, PartialEq, Eq)]
pub enum CanonicalizeResult {
    Changed,
    Unchanged,
}

pub fn canonicalize_op(op: &mut dyn Op) -> CanonicalizeResult {
    let region = op.region();
    if let Some(region) = region {
        let mut changed = CanonicalizeResult::Unchanged;
        for block in region.read().unwrap().blocks() {
            let block = block.read().unwrap();
            let ops = block.ops();
            let ops = ops.read().unwrap();
            let readonly_copy = ops.clone();
            drop(ops);
            for op in readonly_copy.iter() {
                let op: &mut dyn Op = &mut *op.write().unwrap();
                let result = canonicalize_op(op);
                if result == CanonicalizeResult::Changed {
                    changed = CanonicalizeResult::Changed;
                }
            }
        }
        changed
    } else {
        op.canonicalize()
    }
}

pub fn canonicalize(module: &mut ModuleOp) {
    let max_iterations = 16;
    for _ in 0..max_iterations {
        let result = canonicalize_op(module);
        if result == CanonicalizeResult::Unchanged {
            break;
        }
    }
}
