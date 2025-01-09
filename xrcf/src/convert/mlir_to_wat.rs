use crate::convert::apply_rewrites;
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
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<arith::AddiOp>())
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        let op = op.rd();
        let op = op.as_any().downcast_ref::<arith::AddiOp>().unwrap();
        let operation = op.operation().clone();
        let new_op = wat::AddOp::from_operation_arc(operation);
        let new_op = Shared::new(new_op.into());
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

struct FuncLowering;

impl Rewrite for FuncLowering {
    fn name(&self) -> &'static str {
        "convert_mlir_to_wat::FuncLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<func::FuncOp>())
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        let op = op.rd();
        let op = op.as_any().downcast_ref::<func::FuncOp>().unwrap();
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

struct ModuleLowering;

impl Rewrite for ModuleLowering {
    fn name(&self) -> &'static str {
        "convert_mlir_to_wat::ModuleLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<ir::ModuleOp>())
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        let operation = op.rd().operation().clone();
        let new_op = wat::ModuleOp::from_operation_arc(operation);
        let new_op = Shared::new(new_op.into());
        op.rd().replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

/// Convert MLIR to WebAssembly Text format (`.wat`).
pub struct ConvertMLIRToWat;

impl Pass for ConvertMLIRToWat {
    const NAME: &'static str = "convert-mlir-to-wat";
    fn convert(op: Shared<dyn Op>) -> Result<RewriteResult> {
        let rewrites: Vec<&dyn Rewrite> = vec![&AddiLowering, &FuncLowering, &ModuleLowering];
        apply_rewrites(op, &rewrites)
    }
}
