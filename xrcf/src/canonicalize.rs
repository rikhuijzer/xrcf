use crate::convert::apply_rewrites;
use crate::convert::ChangedOp;
use crate::convert::Pass;
use crate::convert::Rewrite;
use crate::convert::RewriteResult;
use crate::ir::Op;
use crate::ir::Users;
use crate::shared::Shared;
use crate::shared::SharedExt;
use anyhow::Result;

pub struct CanonicalizeOp;

impl Rewrite for CanonicalizeOp {
    fn name(&self) -> &'static str {
        "canonicalize::CanonicalizeOp"
    }
    fn parallelizable(&self) -> bool {
        false
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        let result = op.rd().canonicalize();
        Ok(result)
    }
}

pub struct DeadCodeElimination;

impl Rewrite for DeadCodeElimination {
    fn name(&self) -> &'static str {
        "canonicalize::DeadCodeElimination"
    }
    fn parallelizable(&self) -> bool {
        false
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        let readonly = op.clone();
        let readonly = readonly.rd();
        if !readonly.is_pure() {
            return Ok(RewriteResult::Unchanged);
        }
        let operation = readonly.operation().rd();
        let users = operation.users();
        match users {
            Users::HasNoOpResults => Ok(RewriteResult::Unchanged),
            Users::OpOperands(users) => {
                if users.is_empty() {
                    let parent = operation.parent().unwrap();
                    parent.rd().remove(readonly.operation().clone());
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
    fn convert(op: Shared<dyn Op>) -> Result<RewriteResult> {
        let rewrites: Vec<&dyn Rewrite> = vec![&CanonicalizeOp, &DeadCodeElimination];
        apply_rewrites(op, &rewrites)
    }
}
