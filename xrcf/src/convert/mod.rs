//! Conversion logic for the compiler.
//!
//! This module contains conversion passes that can be applied to an IR. By
//! default, this project only implements lowering passes for compilers, but
//! this module can be extended to support other conversions such as "uppering"
//! passes such as used by decompilers.

use crate::ir::spaces;
use crate::ir::Op;
use crate::shared::Shared;
use crate::shared::SharedExt;
use anyhow::Result;
use rayon::prelude::*;
use std::sync::Arc;
use tracing::debug;

mod cf_to_llvm;
mod experimental_to_mlir;
mod func_to_llvm;
mod mlir_to_llvmir;
mod mlir_to_wat;
mod scf_to_cf;

pub use cf_to_llvm::ConvertCFToLLVM;
pub use experimental_to_mlir::ConvertExperimentalToMLIR;
pub use func_to_llvm::ConvertFuncToLLVM;
pub use mlir_to_llvmir::ConvertMLIRToLLVMIR;
pub use mlir_to_wat::ConvertMLIRToWat;
pub use scf_to_cf::ConvertSCFToCF;

pub struct ChangedOp {
    pub op: Shared<dyn Op>,
}

impl ChangedOp {
    pub fn new(op: Shared<dyn Op>) -> Self {
        ChangedOp { op }
    }
}

impl PartialEq for ChangedOp {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.op, &other.op)
    }
}

/// Whether a rewrite changed the IR.
///
/// If a rewrite changes the IR, it returns the changed operation.  Returning
/// the changed operation is required for passes that change the top-level
/// operation.
#[derive(PartialEq)]
pub enum RewriteResult {
    Changed(ChangedOp),
    Unchanged,
}

impl RewriteResult {
    pub fn is_changed(&self) -> Option<&ChangedOp> {
        match self {
            RewriteResult::Changed(op) => Some(op),
            RewriteResult::Unchanged => None,
        }
    }
}

pub trait Rewrite: Send + Sync {
    /// The name of the rewrite; is used for logging.
    fn name(&self) -> &'static str;
    /// Whether the rewrite can be applied in parallel.
    ///
    /// This should only return true if the rewrite can guarantee than any
    /// nested rewrites can also be applied in parallel. For example,
    ///
    /// ```mlir
    /// func.func @f(%arg0: i32) -> i32 {
    ///   %0 = arith.addi %arg0, %arg0 : i32
    ///   return %0 : i32
    /// }
    /// ```
    ///
    /// The rewrite for `func.func` should only be set to true if rewrites on
    /// the nested ops such as `arith.addi` can also be applied in parallel.
    ///
    /// One way in which this for example breaks is when the rewrite for
    /// `arith.addi` would add a global constant to the IR and when this could
    /// clash with other rewrites.
    ///
    /// A drawback of the current approach is that a rewrite needs only one
    /// sub-rewrite that is not parallelizable to make the entire rewrite not
    /// parallelizable. A solution to this could be to make a kind of worklist
    /// of operations that need to be rewritten and then verify that the
    /// worklist only contains parallelizable rewrites. The difficulty here is
    /// that the current implementation may bail out early if a rewrite is
    /// applied. So, I expect that creating the worklist is more work than the
    /// benefit of the parallelization, but it should be benchmarked.
    fn parallelizable(&self) -> bool;
    /// Applies the rewrite to the given operation.
    ///
    /// This method is allowed to mutate the IR.
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult>;
}

fn apply_rewrite_helper(
    root: Shared<dyn Op>,
    rewrite: &dyn Rewrite,
    nested_op: &Shared<dyn Op>,
    indent: i32,
) -> Result<RewriteResult> {
    let indent = indent + 1;
    let result = apply_rewrite(nested_op.clone(), rewrite, indent);
    match result {
        Ok(result) => {
            if result.is_changed().is_some() {
                let root_passthrough = ChangedOp::new(root.clone());
                return Ok(RewriteResult::Changed(root_passthrough));
            }
        }
        Err(e) => {
            return Err(e);
        }
    }
    Ok(RewriteResult::Unchanged)
}

fn apply_rewrite(
    root: Shared<dyn Op>,
    rewrite: &dyn Rewrite,
    indent: i32,
) -> Result<RewriteResult> {
    debug!(
        "{}Matching {} with {}",
        spaces(indent),
        root.clone().rd().name(),
        rewrite.name()
    );
    let root_rewrite = rewrite.rewrite(root.clone())?;
    if root_rewrite.is_changed().is_some() {
        debug!("{}----> Changed", spaces(indent));
        return Ok(root_rewrite);
    }

    fn finder(result: &Result<RewriteResult>) -> bool {
        match result {
            Ok(RewriteResult::Changed(_)) => true,
            Ok(RewriteResult::Unchanged) => false,
            Err(_) => true,
        }
    }
    let ops = root.rd().ops();
    let first_changed = if rewrite.parallelizable() {
        ops.par_iter()
            .map(|nested_op| apply_rewrite_helper(root.clone(), rewrite, nested_op, indent))
            .find_first(finder)
    } else {
        ops.iter()
            .map(|nested_op| apply_rewrite_helper(root.clone(), rewrite, nested_op, indent))
            .find(finder)
    };
    match first_changed {
        Some(result) => match result {
            Ok(RewriteResult::Changed(op)) => Ok(RewriteResult::Changed(op)),
            Ok(RewriteResult::Unchanged) => Ok(RewriteResult::Unchanged),
            Err(e) => Err(e),
        },
        None => Ok(RewriteResult::Unchanged),
    }
}

fn apply_rewrites_helper(
    root: Shared<dyn Op>,
    rewrites: &[&dyn Rewrite],
    indent: i32,
) -> Result<RewriteResult> {
    for rewrite in rewrites {
        let result = apply_rewrite(root.clone(), *rewrite, indent)?;
        if result.is_changed().is_some() {
            return Ok(result);
        }
    }
    Ok(RewriteResult::Unchanged)
}

pub fn apply_rewrites(root: Shared<dyn Op>, rewrites: &[&dyn Rewrite]) -> Result<RewriteResult> {
    let max_iterations = 10240;
    let mut root = root;
    let mut has_changed = false;
    for _ in 0..max_iterations {
        let result = apply_rewrites_helper(root.clone(), rewrites, 0)?;
        match result {
            RewriteResult::Changed(changed) => {
                has_changed = true;
                root = changed.op;
            }
            RewriteResult::Unchanged => {
                if has_changed {
                    let op = ChangedOp::new(root);
                    return Ok(RewriteResult::Changed(op));
                } else {
                    return Ok(result);
                }
            }
        }
    }
    tracing::warn!("Too many rewrite iterations");
    Ok(RewriteResult::Changed(ChangedOp::new(root)))
}

/// A pass is a transformation that can be applied to the IR. MLIR makes a
/// distinction between "translation" and "conversion". A "conversion" stays
/// within MLIr whereas a "translation" can be used to go from MLIR to an
/// external representation. Here, we don't make this distinction.
pub trait Pass {
    const NAME: &'static str;
    fn convert(op: Shared<dyn Op>) -> Result<RewriteResult>;
}

/// Rewrite an operation of type `A` to an operation of type `B`.
///
/// Assumes that the [Operation] from the input can be re-used for the output.
pub fn simple_op_rewrite<A: Op + 'static, B: Op + 'static>(
    op: Shared<dyn Op>,
) -> Result<RewriteResult> {
    let op = op.rd();
    let op = match op.as_any().downcast_ref::<A>() {
        Some(op) => op,
        None => return Ok(RewriteResult::Unchanged),
    };
    let operation = op.operation().clone();
    let new_op = B::from_operation_arc(operation);
    let new_op = Shared::new(new_op.into());
    op.replace(new_op.clone());
    Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
}
