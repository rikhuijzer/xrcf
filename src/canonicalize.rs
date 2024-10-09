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

impl Canonicalize for DeadCodeElimination {
    fn canonicalize(&self, op: &dyn Op) -> CanonicalizeResult {
        // TODO: Figure out whether this op has any uses.
        let operation = op.operation().try_read().unwrap();
        let results = operation.results();
        let results = results.try_read().unwrap();
        for result in results.iter() {
            let result = result.read().unwrap();
            let uses = result.uses();
            if uses.is_empty() {
                println!("DeadCodeElimination: deleting {:?}", result);
            }
        }

        // TODO: If it doesn't, delete it.
        CanonicalizeResult::Unchanged
    }
}

fn canonicalize_op(op: &dyn Op) -> CanonicalizeResult {
    let canonicalizers: Vec<&dyn Canonicalize> = vec![&CanonicalizeOp, &DeadCodeElimination];
    let mut changed = CanonicalizeResult::Unchanged;
    for canonicalizer in &canonicalizers {
        // Determine ops here because `canonicalizer.canonicalize` may delete an op.
        let ops = op.ops();
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
