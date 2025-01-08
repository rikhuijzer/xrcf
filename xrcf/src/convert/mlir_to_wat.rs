use crate::convert::apply_rewrites;
use crate::convert::Pass;
use crate::convert::Rewrite;
use crate::convert::RewriteResult;
use crate::ir::Op;
use crate::shared::Shared;
use anyhow::Result;

pub struct ConvertMLIRToWat;

impl Pass for ConvertMLIRToWat {
    const NAME: &'static str = "convert-mlir-to-wat";
    fn convert(op: Shared<dyn Op>) -> Result<RewriteResult> {
        let rewrites: Vec<&dyn Rewrite> = vec![];
        apply_rewrites(op, &rewrites)
    }
}
