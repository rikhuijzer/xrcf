use crate::ir::ModuleOp;
use crate::ir::Op;
use crate::ir::Users;

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
        let operation = op.operation().try_read().unwrap();
        let users = operation.users();
        match users {
            Users::HasNoOpResults => CanonicalizeResult::Unchanged,
            Users::OpOperands(users) => {
                if users.is_empty() {
                    let parent = operation.parent();
                    let parent = parent.unwrap();
                    let parent = parent.try_read().unwrap();
                    parent.remove(op.operation().clone());
                    CanonicalizeResult::Changed
                } else {
                    CanonicalizeResult::Unchanged
                }
            }
        }
    }
}

fn canonicalize_op(op: &dyn Op) -> CanonicalizeResult {
    let canonicalizers: Vec<&dyn Canonicalize> = vec![&CanonicalizeOp, &DeadCodeElimination];
    for canonicalizer in &canonicalizers {
        // Determine ops here because `canonicalizer.canonicalize` may delete an op.
        let ops = op.ops();
        for nested_op in ops.iter() {
            let nested_op = nested_op.read().unwrap();
            let result = canonicalize_op(&*nested_op);
            if result == CanonicalizeResult::Changed {
                return result;
            }
        }
        let result = canonicalizer.canonicalize(op);
        if result == CanonicalizeResult::Changed {
            return result;
        }
    }
    CanonicalizeResult::Unchanged
}

pub fn canonicalize(op: &ModuleOp) {
    let max_iterations = 16;
    for _ in 0..max_iterations {
        println!("Canonicalizing");
        let result = canonicalize_op(op);
        if result == CanonicalizeResult::Unchanged {
            break;
        }
    }
}
