use crate::ir::ModuleOp;
use crate::ir::Op;

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
            let block = block.write().unwrap();
            for op in block.ops().write().unwrap().iter_mut() {
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
