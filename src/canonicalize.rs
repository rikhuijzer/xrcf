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
        op.canonicalize()
    }
}

struct DeadCodeElimination;

impl DeadCodeElimination {
    fn eliminate(&self, op: &dyn Op) -> CanonicalizeResult {
        todo!()
    }
}

impl Canonicalize for DeadCodeElimination {
    fn canonicalize(&self, op: &dyn Op) -> CanonicalizeResult {
        CanonicalizeResult::Unchanged
    }
}

fn canonicalize_op(op: &dyn Op) -> CanonicalizeResult {
    let canonicalizers: Vec<&dyn Canonicalize> = vec![&CanonicalizeOp]; // &DeadCodeElimination];
    let mut changed = CanonicalizeResult::Unchanged;
    let ops = op.ops();
    for canonicalizer in &canonicalizers {
        for nested_op in ops.iter() {
            let nested_op = nested_op.read().unwrap();
            let result = canonicalize_op(&*nested_op);
            if result == CanonicalizeResult::Changed {
                changed = result;
            }
            let result = canonicalizer.canonicalize(op);
            if result == CanonicalizeResult::Changed {
                changed = result;
            }
        }
        let result = canonicalizer.canonicalize(op);
        if result == CanonicalizeResult::Changed {
            changed = result;
        }
    }
    changed
}

pub fn canonicalize(op: &ModuleOp) {
    let max_iterations = 16;
    for _ in 0..max_iterations {
        let result = canonicalize_op(op);
        if result == CanonicalizeResult::Unchanged {
            break;
        }
    }
}
