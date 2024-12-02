use crate::convert::apply_rewrites;
use crate::convert::ChangedOp;
use crate::convert::Pass;
use crate::convert::Rewrite;
use crate::convert::RewriteResult;
use crate::ir::GuardedBlock;
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
    fn is_match(&self, _op: &dyn Op) -> Result<bool> {
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
    fn is_match(&self, _op: &dyn Op) -> Result<bool> {
        Ok(true)
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let readonly = op.clone();
        let readonly = readonly.try_read().unwrap();
        if !readonly.is_pure() {
            return Ok(RewriteResult::Unchanged);
        }
        let operation = readonly.operation().try_read().unwrap();
        let users = operation.users();
        match users {
            Users::HasNoOpResults => Ok(RewriteResult::Unchanged),
            Users::OpOperands(users) => {
                if users.is_empty() {
                    let parent = operation.parent().unwrap();
                    parent.remove(readonly.operation().clone());
                    Ok(RewriteResult::Changed(ChangedOp::new(op.clone())))
                } else {
                    Ok(RewriteResult::Unchanged)
                }
            }
        }
    }
}

pub struct Canonicalize;

impl Pass for Canonicalize {
    const NAME: &'static str = "canonicalize";
    fn convert(op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let rewrites: Vec<&dyn Rewrite> = vec![&CanonicalizeOp, &DeadCodeElimination];
        apply_rewrites(op, &rewrites)
    }
}
