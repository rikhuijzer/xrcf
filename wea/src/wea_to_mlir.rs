use anyhow::Result;
use xrcf::convert::apply_rewrites;
use xrcf::convert::ChangedOp;
use xrcf::convert::Pass;
use xrcf::convert::Rewrite;
use xrcf::convert::RewriteResult;
use xrcf::dialect::func;
use xrcf::ir::Op;
use xrcf::shared::Shared;
use xrcf::shared::SharedExt;

struct FuncLowering;

impl Rewrite for FuncLowering {
    fn name(&self) -> &'static str {
        "func-lowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<func::FuncOp>())
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        let op = op.rd();
        let op = op.as_any().downcast_ref::<func::FuncOp>().unwrap();
        let operation = op.operation().clone();
        let new_op = func::FuncOp::from_operation_arc(operation);
        let new_op = Shared::new(new_op.into());
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

#[allow(dead_code)]
pub struct ConvertWeaToMLIR;

impl Pass for ConvertWeaToMLIR {
    const NAME: &'static str = "convert-wea-to-mlir";
    fn convert(op: Shared<dyn Op>) -> Result<RewriteResult> {
        let rewrites: Vec<&dyn Rewrite> = vec![&FuncLowering];
        apply_rewrites(op, &rewrites)
    }
}
