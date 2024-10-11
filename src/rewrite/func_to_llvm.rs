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
        Ok(op.operation().read().unwrap().name() == func::FuncOp::operation_name())
    }
    fn rewrite(&self, op: &dyn Op) -> Result<RewriteResult> {
        {
            let operation = op.operation();
            let mut operation = operation.try_write().unwrap();
            let name = operation.name();
            assert!(name == func::FuncOp::operation_name());
            operation.set_name(llvmir::FuncOp::operation_name());
        }

        let operation = op.operation().clone();
        let lowered = llvmir::FuncOp::from_operation(op.operation().clone())?;
        println!("here");
        op.replace(Arc::new(RwLock::new(lowered)));
        println!("here2");

        Ok(RewriteResult::Changed)
    }
}

struct ConstantOpLowering;

impl Rewrite for ConstantOpLowering {
    fn is_match(&self, _op: &dyn Op) -> Result<bool> {
        Ok(true)
    }
    fn rewrite(&self, _op: &dyn Op) -> Result<RewriteResult> {
        Ok(RewriteResult::Unchanged)
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
