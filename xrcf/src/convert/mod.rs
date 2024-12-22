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
mod scf_to_cf;

pub use cf_to_llvm::ConvertCFToLLVM;
pub use experimental_to_mlir::ConvertExperimentalToMLIR;
pub use func_to_llvm::ConvertFuncToLLVM;
pub use mlir_to_llvmir::ConvertMLIRToLLVMIR;
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
    /// Returns true if the rewrite can be applied to the given operation.
    ///
    /// This method is not allowed to mutate the IR.
    ///
    /// Note that this implementation usually will look like
    /// ```mlir
    /// Ok(op.as_any().is::<MyOp>())
    /// ```
    /// If weird behavior is encountered, ensure that the new type has set the
    /// correct operation name. Otherwise, an object might look like is a
    /// different type than it really is.
    fn is_match(&self, op: &dyn Op) -> Result<bool>;
    /// Applies the rewrite to the given operation.
    ///
    /// This method is allowed to mutate the IR.
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult>;
}

fn apply_rewrites_core(
    root: Shared<dyn Op>,
    rewrites: &[&dyn Rewrite],
    indent: i32,
) -> Result<RewriteResult> {
    for rewrite in rewrites {
        // Determine ops here because `rewrite` may delete an op.
        for op in root.rd().ops().iter() {
            let indent = indent + 1;
            let result = apply_rewrites_core(op.clone(), rewrites, indent)?;
            if result.is_changed().is_some() {
                let root_passthrough = ChangedOp::new(root.clone());
                let root_passthrough = RewriteResult::Changed(root_passthrough);
                return Ok(root_passthrough);
            }
        }
        debug!(
            "{}Matching {} with {}",
            spaces(indent),
            root.rd().name(),
            rewrite.name()
        );
        if rewrite.is_match(&*root.rd())? {
            debug!("{}--> Success", spaces(indent));
            let root_rewrite = rewrite.rewrite(root.clone())?;
            if root_rewrite.is_changed().is_some() {
                debug!("{}----> Changed", spaces(indent));
                return Ok(root_rewrite);
            }
        }
    }
    Ok(RewriteResult::Unchanged)
}

pub fn apply_rewrites(root: Shared<dyn Op>, rewrites: &[&dyn Rewrite]) -> Result<RewriteResult> {
    let max_iterations = 1024;
    let mut root = root;
    let mut has_changed = false;
    for _ in 0..max_iterations {
        let result = apply_rewrites_core(root.clone(), rewrites, 0)?;
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
