//! Conversion logic for the compiler.
//!
//! This module contains conversion passes that can be applied to an IR. By
//! default, this project only implements lowering passes for compilers, but
//! this module can be extended to support other conversions such as "uppering"
//! passes such as used by decompilers.

use crate::ir::spaces;
use crate::ir::Op;
use anyhow::Result;
use std::sync::Arc;
use std::sync::RwLock;
use tracing::debug;

mod func_to_llvm;
mod mlir_to_llvmir;
mod unstable_to_mlir;

pub use func_to_llvm::ConvertFuncToLLVM;
pub use mlir_to_llvmir::ConvertMLIRToLLVMIR;
pub use unstable_to_mlir::ConvertUnstableToMLIR;

pub struct ChangedOp(pub Arc<RwLock<dyn Op>>);

impl ChangedOp {
    pub fn new(op: Arc<RwLock<dyn Op>>) -> Self {
        ChangedOp(op)
    }
}

impl PartialEq for ChangedOp {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
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
    pub fn is_changed(&self) -> bool {
        matches!(self, RewriteResult::Changed(_))
    }
}

pub trait Rewrite {
    fn name(&self) -> &'static str;
    /// Returns true if the rewrite can be applied to the given operation.
    ///
    /// This method is not allowed to mutate the IR.
    fn is_match(&self, op: &dyn Op) -> Result<bool>;
    /// Applies the rewrite to the given operation.
    ///
    /// This method is allowed to mutate the IR.
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult>;
}

fn apply_rewrites_helper(
    root: Arc<RwLock<dyn Op>>,
    rewrites: &[&dyn Rewrite],
    indent: i32,
) -> Result<RewriteResult> {
    let ops = root.try_read().unwrap().ops();
    for rewrite in rewrites {
        // Determine ops here because `rewrite` may delete an op.
        for nested_op in ops.iter() {
            let indent = indent + 1;
            let result = apply_rewrites_helper(nested_op.clone(), rewrites, indent)?;
            if result.is_changed() {
                let root_passthrough = ChangedOp::new(root.clone());
                let root_passthrough = RewriteResult::Changed(root_passthrough);
                return Ok(root_passthrough);
            }
        }
        debug!(
            "{}Matching {} with {}",
            spaces(indent),
            root.clone().try_read().unwrap().name(),
            rewrite.name()
        );
        let root_read = root.clone();
        let root_read = root_read.try_read().unwrap();
        if rewrite.is_match(&*root_read)? {
            debug!("{}--> Success", spaces(indent));
            let root_rewrite = rewrite.rewrite(root.clone())?;
            if root_rewrite.is_changed() {
                debug!("{}----> Changed", spaces(indent));
                return Ok(root_rewrite);
            }
        }
    }
    Ok(RewriteResult::Unchanged)
}

pub fn apply_rewrites(
    root: Arc<RwLock<dyn Op>>,
    rewrites: &[&dyn Rewrite],
) -> Result<RewriteResult> {
    let max_iterations = 16;
    let mut root = root;
    let mut has_changed = false;
    for _ in 0..max_iterations {
        let result = apply_rewrites_helper(root.clone(), rewrites, 0)?;
        match result {
            RewriteResult::Changed(changed) => {
                has_changed = true;
                root = changed.0;
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
    tracing::warn!("too many rewrite iterations");
    Ok(RewriteResult::Changed(ChangedOp::new(root)))
}

/// A pass is a transformation that can be applied to the IR. MLIR makes a
/// distinction between "translation" and "conversion". A "conversion" stays
/// within MLIr whereas a "translation" can be used to go from MLIR to an
/// external representation. Here, we don't make this distinction.
pub trait Pass {
    const NAME: &'static str;
    fn convert(op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult>;
}
