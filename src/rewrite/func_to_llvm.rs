use crate::dialect::arith;
use crate::dialect::func;
use crate::dialect::llvmir;
use crate::ir::Op;
use crate::rewrite::apply_rewrites;
use crate::rewrite::Rewrite;
use crate::rewrite::RewriteResult;
use crate::Pass;
use anyhow::Result;
use std::sync::Arc;
use std::sync::RwLock;

struct FuncLowering;

impl Rewrite for FuncLowering {
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().downcast_ref::<func::FuncOp>().is_some())
    }
    fn rewrite(&self, op: &dyn Op) -> Result<RewriteResult> {
        let op = op.as_any().downcast_ref::<func::FuncOp>().unwrap();
        {
            let operation = op.operation();
            let mut operation = operation.try_write().unwrap();
            let name = operation.name();
            assert!(name == func::FuncOp::operation_name());
            operation.set_name(llvmir::FuncOp::operation_name());

            let parent = operation.parent();
            assert!(
                parent.is_some(),
                "maybe func parent was not set during parsing?"
            );
        }

        let mut lowered = llvmir::FuncOp::from_operation(op.operation().clone())?;
        lowered.set_identifier(op.identifier().to_string());
        op.replace(Arc::new(RwLock::new(lowered)));

        Ok(RewriteResult::Changed)
    }
}

struct ConstantOpLowering;

impl Rewrite for ConstantOpLowering {
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().downcast_ref::<arith::ConstantOp>().is_some())
    }
    fn rewrite(&self, op: &dyn Op) -> Result<RewriteResult> {
        let op = op.as_any().downcast_ref::<arith::ConstantOp>().unwrap();
        let lowered = llvmir::ConstantOp::from_operation(op.operation().clone())?;
        op.replace(Arc::new(RwLock::new(lowered)));

        Ok(RewriteResult::Changed)
    }
}

pub struct ConvertFuncToLLVM;

impl Pass for ConvertFuncToLLVM {
    fn name() -> &'static str {
        "convert-func-to-llvm"
    }
    fn convert(op: &dyn Op) -> Result<()> {
        let rewrites: Vec<&dyn Rewrite> = vec![&FuncLowering, &ConstantOpLowering];
        apply_rewrites(op, &rewrites)?;
        Ok(())
    }
}
