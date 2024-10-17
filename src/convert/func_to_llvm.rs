use crate::convert::apply_rewrites;
use crate::convert::ChangedOp;
use crate::convert::Rewrite;
use crate::convert::RewriteResult;
use crate::dialect::arith;
use crate::dialect::func;
use crate::dialect::llvmir;
use crate::ir::Op;
use crate::Pass;
use anyhow::Result;
use std::sync::Arc;
use std::sync::RwLock;

struct FuncLowering;

impl Rewrite for FuncLowering {
    fn name(&self) -> &'static str {
        "func_to_llvm::FuncLowering"
    }
    fn is_match(&self, op: Arc<RwLock<dyn Op>>) -> Result<bool> {
        Ok(op
            .try_read()
            .unwrap()
            .as_any()
            .downcast_ref::<func::FuncOp>()
            .is_some())
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let op = op.try_read().unwrap();
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
        let new_op = Arc::new(RwLock::new(lowered));
        op.replace(new_op.clone());

        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

struct ConstantOpLowering;

impl Rewrite for ConstantOpLowering {
    fn name(&self) -> &'static str {
        "func_to_llvm::ConstantOpLowering"
    }
    fn is_match(&self, op: Arc<RwLock<dyn Op>>) -> Result<bool> {
        Ok(op
            .try_read()
            .unwrap()
            .as_any()
            .downcast_ref::<arith::ConstantOp>()
            .is_some())
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let op = op.try_read().unwrap();
        let op = op.as_any().downcast_ref::<arith::ConstantOp>().unwrap();
        let lowered = llvmir::ConstantOp::from_operation(op.operation().clone())?;
        let new_op = Arc::new(RwLock::new(lowered));
        op.replace(new_op.clone());

        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

struct ReturnLowering;

impl Rewrite for ReturnLowering {
    fn name(&self) -> &'static str {
        "func_to_llvm::ReturnLowering"
    }
    fn is_match(&self, op: Arc<RwLock<dyn Op>>) -> Result<bool> {
        Ok(op
            .try_read()
            .unwrap()
            .as_any()
            .downcast_ref::<func::ReturnOp>()
            .is_some())
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let op = op.try_read().unwrap();
        let op = op.as_any().downcast_ref::<func::ReturnOp>().unwrap();
        let lowered = llvmir::ReturnOp::from_operation(op.operation().clone())?;
        let new_op = Arc::new(RwLock::new(lowered));
        op.replace(new_op.clone());

        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

pub struct ConvertFuncToLLVM;

impl Pass for ConvertFuncToLLVM {
    fn name() -> &'static str {
        "func_to_llvm::ConvertFuncToLLVM"
    }
    fn convert(op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let rewrites: Vec<&dyn Rewrite> = vec![&FuncLowering, &ConstantOpLowering, &ReturnLowering];
        apply_rewrites(op, &rewrites)
    }
}
