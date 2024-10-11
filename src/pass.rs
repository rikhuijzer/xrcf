use crate::ir::Op;
use anyhow::Result;

pub trait Pass {
    fn name() -> &'static str;
    fn convert(op: &dyn Op) -> Result<()>;
}
