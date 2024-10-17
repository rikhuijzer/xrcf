use crate::convert::ChangedOp;
use crate::convert::Rewrite;
use crate::convert::RewriteResult;
use crate::ir::Op;
use crate::ir::Users;
use anyhow::Result;
use std::sync::Arc;
use std::sync::RwLock;

pub struct CanonicalizeOp;

impl Rewrite for CanonicalizeOp {
    fn name(&self) -> &'static str {
        "canonicalize::CanonicalizeOp"
    }
    fn is_match(&self, _op: Arc<RwLock<dyn Op>>) -> Result<bool> {
        Ok(true)
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let op = op.try_read().unwrap();
        let result = op.canonicalize();
        Ok(result)
    }
}

pub struct DeadCodeElimination;

impl Rewrite for DeadCodeElimination {
    fn name(&self) -> &'static str {
        "canonicalize::DeadCodeElimination"
    }
    fn is_match(&self, _op: Arc<RwLock<dyn Op>>) -> Result<bool> {
        Ok(true)
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let readonly = op.clone();
        let readonly = readonly.try_read().unwrap();
        let operation = readonly.operation().try_read().unwrap();
        let users = operation.users();
        match users {
            Users::HasNoOpResults => Ok(RewriteResult::Unchanged),
            Users::OpOperands(users) => {
                if users.is_empty() {
                    let parent = operation.parent();
                    let parent = parent.unwrap();
                    let parent = parent.try_read().unwrap();
                    parent.remove(readonly.operation().clone());
                    Ok(RewriteResult::Changed(ChangedOp::new(op.clone())))
                } else {
                    Ok(RewriteResult::Unchanged)
                }
            }
        }
    }
}
