use crate::ir::ModuleOp;
use crate::ir::Op;

#[derive(Debug, PartialEq, Eq)]
pub enum CanonicalizeResult {
    Changed,
    Unchanged,
}

pub trait Canonicalize {
    fn canonicalize(&self, op: &dyn Op) -> CanonicalizeResult;
}

struct CanonicalizeOp;

impl Canonicalize for CanonicalizeOp {
    fn canonicalize(&self, op: &dyn Op) -> CanonicalizeResult {
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
                    let op: &dyn Op = &*op.read().unwrap();
                    let result = CanonicalizeOp.canonicalize(op);
                    if result == CanonicalizeResult::Changed {
                        changed = result;
                    }
                }
            }
            changed
        } else {
            op.canonicalize()
        }
    }
}

struct DeadCodeElimination;

impl Canonicalize for DeadCodeElimination {
    fn canonicalize(&self, op: &dyn Op) -> CanonicalizeResult {
        CanonicalizeResult::Unchanged
    }
}

pub fn canonicalize(module: &mut ModuleOp) {
    let max_iterations = 16;
    for _ in 0..max_iterations {
        let canonicalizers: Vec<&dyn Canonicalize> = vec![&CanonicalizeOp, &DeadCodeElimination];
        let mut changed = false;
        for canonicalizer in canonicalizers {
            let result = canonicalizer.canonicalize(module);
            if result == CanonicalizeResult::Changed {
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}
