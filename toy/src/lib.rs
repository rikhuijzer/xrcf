mod op;
mod toy_to_mlir;

use anyhow::Result;
use std::sync::Arc;
use std::sync::RwLock;
use toy_to_mlir::ConvertToyToMLIR;
pub use xrcf::compile;
use xrcf::convert::RewriteResult;
use xrcf::ir::Block;
use xrcf::ir::Op;
use xrcf::parser::ParserDispatch;
use xrcf::CompilerDispatch;
use xrcf::DefaultCompilerDispatch;
use xrcf::Parser;
use xrcf::Pass;

pub struct ToyCompilerDispatch;

impl CompilerDispatch for ToyCompilerDispatch {
    fn dispatch(op: Arc<RwLock<dyn Op>>, pass: &str) -> Result<RewriteResult> {
        let result = DefaultCompilerDispatch::dispatch(op.clone(), pass)?;
        if let RewriteResult::Changed(_) = result {
            return Ok(result);
        }
        match pass {
            ConvertToyToMLIR::NAME => ConvertToyToMLIR::convert(op),
            _ => return Err(anyhow::anyhow!("Unknown pass: {}", pass)),
        }
    }
}

pub struct ToyParserDispatch;

impl ParserDispatch for ToyParserDispatch {
    fn dispatch_parse(
        name: String,
        parser: &mut Parser<Self>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        match name.as_str() {
            "def" => todo!(),
            "laat" => todo!(),
            "retour" => todo!(),
            _ => return Err(anyhow::anyhow!("Unknown op: {}", name)),
        }
    }
    fn parse_op(
        parser: &mut Parser<Self>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let name = parser.peek();
        let name = name.lexeme.clone();
        Self::dispatch_parse(name, parser, parent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use indoc::indoc;

    #[test]
    fn test_convert_toy_to_mlir() {
        let src = indoc! {"
        def main() -> i32 {
          laat x = 42
          retour x
        }
        "};
        let src = src.trim();
        let expected = indoc! {"
        module {
          func.func @main() -> i32 {
            %0 = arith.constant 42 : i32
            return %0 : i32
          }
        }
        "};
        let _expected = expected.trim();
        let module = Parser::<ToyParserDispatch>::parse(src).unwrap();
        let module = module.try_read().unwrap();
        let parsed = module.to_string();
        assert_eq!(src, parsed);
    }
}
