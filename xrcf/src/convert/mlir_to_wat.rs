use crate::convert::apply_rewrites;
use crate::convert::simple_op_rewrite;
use crate::convert::ChangedOp;
use crate::convert::Pass;
use crate::convert::Rewrite;
use crate::convert::RewriteResult;
use crate::dialect::arith;
use crate::dialect::func;
use crate::dialect::func::Func;
use crate::ir;
use crate::ir::Op;
use crate::shared::Shared;
use crate::shared::SharedExt;
use crate::targ3t::wat;
use crate::targ3t::wat::SymVisibility;
use anyhow::Result;

struct AddiLowering;

impl Rewrite for AddiLowering {
    fn name(&self) -> &'static str {
        "convert_mlir_to_wat::AddiLowering"
    }
    fn parallelizable(&self) -> bool {
        true
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        simple_op_rewrite::<arith::AddiOp, wat::AddOp>(op)
    }
}

struct FuncLowering;

impl Rewrite for FuncLowering {
    fn name(&self) -> &'static str {
        "convert_mlir_to_wat::FuncLowering"
    }
    fn parallelizable(&self) -> bool {
        true
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        let op = op.rd();
        let op = match op.as_any().downcast_ref::<func::FuncOp>() {
            Some(op) => op,
            None => return Ok(RewriteResult::Unchanged),
        };
        let operation = op.operation().clone();
        let mut new_op = wat::FuncOp::from_operation_arc(operation);
        if op.sym_visibility().is_none() {
            new_op.sym_visibility = Some(SymVisibility::Public);
        }
        if let Some(identifier) = op.identifier() {
            new_op.set_identifier(Some(identifier));
        }
        let new_op = Shared::new(new_op.into());
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

struct SubiLowering;

impl Rewrite for SubiLowering {
    fn name(&self) -> &'static str {
        "convert_mlir_to_wat::SubiLowering"
    }
    fn parallelizable(&self) -> bool {
        true
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        simple_op_rewrite::<arith::SubiOp, wat::SubOp>(op)
    }
}

struct ModuleLowering;

impl Rewrite for ModuleLowering {
    fn name(&self) -> &'static str {
        "convert_mlir_to_wat::ModuleLowering"
    }
    fn parallelizable(&self) -> bool {
        true
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        simple_op_rewrite::<ir::ModuleOp, wat::ModuleOp>(op)
    }
}

struct ReturnLowering;

impl Rewrite for ReturnLowering {
    fn name(&self) -> &'static str {
        "convert_mlir_to_wat::ReturnLowering"
    }
    fn parallelizable(&self) -> bool {
        true
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        simple_op_rewrite::<func::ReturnOp, wat::ReturnOp>(op)
    }
}

/// Convert MLIR to WebAssembly Text format (`.wat`).
pub struct ConvertMLIRToWat;

impl Pass for ConvertMLIRToWat {
    const NAME: &'static str = "convert-mlir-to-wat";
    fn convert(op: Shared<dyn Op>) -> Result<RewriteResult> {
        let rewrites: Vec<&dyn Rewrite> = vec![
            &AddiLowering,
            &FuncLowering,
            &SubiLowering,
            &ModuleLowering,
            &ReturnLowering,
        ];
        apply_rewrites(op, &rewrites)
    }
}
