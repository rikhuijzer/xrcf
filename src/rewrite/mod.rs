use crate::ir::Op;
use anyhow::Result;

mod func_to_llvm;

pub use func_to_llvm::ConvertFuncToLLVM;

#[derive(Debug, PartialEq, Eq)]
pub enum RewriteResult {
    Changed,
    Unchanged,
}

pub trait Rewrite {
    /// Returns true if the rewrite can be applied to the given operation.
    /// This method is not allowed to mutate the IR.
    fn is_match(&self, op: &dyn Op) -> Result<bool>;
    /// Applies the rewrite to the given operation.
    /// This method is allowed to mutate the IR.
    fn rewrite(&self, op: &dyn Op) -> Result<RewriteResult>;
}

fn apply_rewrites_helper(op: &dyn Op, rewrites: &[&dyn Rewrite]) -> Result<RewriteResult> {
    for rewrite in rewrites {
        // Determine ops here because `rewrite` may delete an op.
        let ops = op.ops();
        for nested_op in ops.iter() {
            let nested_op = nested_op.try_read().unwrap();
            let result = apply_rewrites_helper(&*nested_op, rewrites)?;
            if result == RewriteResult::Changed {
                return Ok(result);
            }
        }
        if rewrite.is_match(op)? {
            let result = rewrite.rewrite(op)?;
            if result == RewriteResult::Changed {
                return Ok(result);
            }
        }
    }
    Ok(RewriteResult::Unchanged)
}

pub fn apply_rewrites(op: &mut dyn Op, rewrites: &[&dyn Rewrite]) -> Result<()> {
    let max_iterations = 16;
    for _ in 0..max_iterations {
        let result = apply_rewrites_helper(op, rewrites)?;
        if result == RewriteResult::Unchanged {
            return Ok(());
        }
    }
    anyhow::bail!("too many rewrite iterations");
}
