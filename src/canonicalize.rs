use crate::ir::Op;
use crate::ir::Users;
use crate::rewrite::Rewrite;
use crate::rewrite::RewriteResult;
use anyhow::Result;

pub struct CanonicalizeOp;

impl Rewrite for CanonicalizeOp {
    fn is_match(&self, _op: &dyn Op) -> Result<bool> {
        Ok(true)
    }
    fn rewrite(&self, op: &dyn Op) -> Result<RewriteResult> {
        let result = op.canonicalize();
        Ok(result)
    }
}

pub struct DeadCodeElimination;

impl Rewrite for DeadCodeElimination {
    fn is_match(&self, _op: &dyn Op) -> Result<bool> {
        Ok(true)
    }
    fn rewrite(&self, op: &dyn Op) -> Result<RewriteResult> {
        let operation = op.operation().try_read().unwrap();
        let users = operation.users();
        match users {
            Users::HasNoOpResults => Ok(RewriteResult::Unchanged),
            Users::OpOperands(users) => {
                if users.is_empty() {
                    let parent = operation.parent();
                    let parent = parent.unwrap();
                    let parent = parent.try_read().unwrap();
                    parent.remove(op.operation().clone());
                    Ok(RewriteResult::Changed)
                } else {
                    Ok(RewriteResult::Unchanged)
                }
            }
        }
    }
}
