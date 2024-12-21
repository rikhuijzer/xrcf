use crate::convert::apply_rewrites;
use crate::convert::ChangedOp;
use crate::convert::Pass;
use crate::convert::Rewrite;
use crate::convert::RewriteResult;
use crate::dialect::arith;
use crate::dialect::func;
use crate::dialect::func::Call;
use crate::dialect::func::Func;
use crate::dialect::llvm;
use crate::ir::Op;
use crate::shared::Shared;
use crate::shared::SharedExt;
use anyhow::Result;

struct AddLowering;

impl Rewrite for AddLowering {
    fn name(&self) -> &'static str {
        "func_to_llvm::AddOpLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<arith::AddiOp>())
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        let op = op.rd();
        let lowered = llvm::AddOp::from_operation_arc(op.operation().clone());
        let new_op = Shared::new(lowered.into());
        op.replace(new_op.clone());

        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

struct CallLowering;

impl Rewrite for CallLowering {
    fn name(&self) -> &'static str {
        "func_to_llvm::CallLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<func::CallOp>())
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        let op = op.rd();
        let op = op.as_any().downcast_ref::<func::CallOp>().unwrap();
        let mut new_op = llvm::CallOp::from_operation_arc(op.operation().clone());
        new_op.set_identifier(op.identifier().unwrap());
        let new_op = Shared::new(new_op.into());
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
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        let op = op.rd();
        let lowered = llvm::ConstantOp::from_operation_arc(op.operation().clone());
        let new_op = Shared::new(lowered.into());
        op.replace(new_op.clone());

        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

struct FuncLowering;

impl Rewrite for FuncLowering {
    fn name(&self) -> &'static str {
        "func_to_llvm::FuncLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<func::FuncOp>())
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        let op = op.rd();
        let op = op.as_any().downcast_ref::<func::FuncOp>().unwrap();
        let operation = op.operation();
        {
            let name = operation.rd().name();
            assert!(name == func::FuncOp::operation_name());
            operation.wr().set_name(name.clone());

            let parent = operation.rd().parent();
            assert!(
                parent.is_some(),
                "maybe parent was not set during parsing for {name}?",
            );
        }

        let mut new_op = llvm::FuncOp::from_operation_arc(operation.clone());
        new_op.set_identifier(op.identifier().unwrap());
        new_op.set_sym_visibility(op.sym_visibility().clone());
        let new_op = Shared::new(new_op.into());
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
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        let op = op.rd();
        let op = op.as_any().downcast_ref::<func::ReturnOp>().unwrap();
        let lowered = llvm::ReturnOp::from_operation_arc(op.operation().clone());
        let new_op = Shared::new(lowered.into());
        op.replace(new_op.clone());

        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

pub struct ConvertFuncToLLVM;

impl Pass for ConvertFuncToLLVM {
    const NAME: &'static str = "convert-func-to-llvm";
    fn convert(op: Shared<dyn Op>) -> Result<RewriteResult> {
        let rewrites: Vec<&dyn Rewrite> = vec![
            &AddLowering,
            &CallLowering,
            &ConstantOpLowering,
            &FuncLowering,
            &ReturnLowering,
        ];
        apply_rewrites(op, &rewrites)
    }
}
