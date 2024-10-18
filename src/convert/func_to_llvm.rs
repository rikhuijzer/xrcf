use crate::convert::apply_rewrites;
use crate::convert::ChangedOp;
use crate::convert::Rewrite;
use crate::convert::RewriteResult;
use crate::dialect::arith;
use crate::dialect::func;
use crate::dialect::func::Func;
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
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<func::FuncOp>())
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
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<arith::ConstantOp>())
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let op = op.try_read().unwrap();
        let lowered = llvmir::ConstantOp::from_operation(op.operation().clone())?;
        let new_op = Arc::new(RwLock::new(lowered));
        op.replace(new_op.clone());

        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

struct AddOpLowering;

impl Rewrite for AddOpLowering {
    fn name(&self) -> &'static str {
        "func_to_llvm::AddOpLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<arith::AddiOp>())
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let op = op.try_read().unwrap();
        let lowered = llvmir::AddOp::from_operation(op.operation().clone())?;
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
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<func::ReturnOp>())
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
        let rewrites: Vec<&dyn Rewrite> = vec![
            &FuncLowering,
            &ConstantOpLowering,
            &AddOpLowering,
            &ReturnLowering,
        ];
        apply_rewrites(op, &rewrites)
    }
}
