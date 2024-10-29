use crate::python;
use crate::python_to_mlir::ConvertPythonToMLIR;
use anyhow::Result;
use std::sync::Arc;
use std::sync::RwLock;
use xrcf::convert::Pass;
use xrcf::convert::RewriteResult;
use xrcf::ir::Block;
use xrcf::ir::Op;
use xrcf::ir::Type;
use xrcf::parser::default_dispatch;
use xrcf::parser::Parse;
use xrcf::parser::Parser;
use xrcf::parser::ParserDispatch;
use xrcf::parser::TokenKind;
use xrcf::transform;
use xrcf::DefaultTransformDispatch;
use xrcf::Passes;
use xrcf::SinglePass;
use xrcf::TransformDispatch;

struct PythonParserDispatch;

impl ParserDispatch for PythonParserDispatch {
    fn parse_op(
        parser: &mut Parser<Self>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let name = if parser.peek_n(1).unwrap().kind == TokenKind::Equal {
            // Ignore result name and '=' (e.g., `x = <op name>`).
            parser.peek_n(2).unwrap().clone()
        } else {
            // Ignore nothing (e.g., `<op name> x, y`).
            parser.peek().clone()
        };
        match name.lexeme.clone().as_str() {
            "def" => <python::FuncOp as Parse>::op(parser, parent),
            _ => default_dispatch(name, parser, parent),
        }
    }
    fn parse_type(_parser: &mut Parser<Self>) -> Result<Arc<RwLock<dyn Type>>> {
        todo!()
    }
}

struct PythonTransformDispatch;

impl TransformDispatch for PythonTransformDispatch {
    fn dispatch(op: Arc<RwLock<dyn Op>>, pass: &SinglePass) -> Result<RewriteResult> {
        match pass.to_string().as_str() {
            "convert-python-to-mlir" => ConvertPythonToMLIR::convert(op),
            _ => DefaultTransformDispatch::dispatch(op, pass),
        }
    }
}

pub fn parse_and_transform(src: &str, passes: &Passes) -> Result<RewriteResult> {
    let op = Parser::<PythonParserDispatch>::parse(src)?;
    let result = transform::<PythonTransformDispatch>(op, passes)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use indoc::indoc;

    #[test]
    fn test_transform() {
        let src = indoc! {r#"
        func.func @main() -> i32 {
            %0 = arith.constant 0 : i32
            return %0 : i32
        }
        "#};
        let passes = vec!["--convert-func-to-llvm", "--convert-mlir-to-llvmir"];
        let passes = Passes::from_vec(passes);
        let result = parse_and_transform(src, &passes).unwrap();
        let result = match result {
            RewriteResult::Changed(op) => {
                let op = op.0.try_read().unwrap();
                op.to_string()
            }
            RewriteResult::Unchanged => {
                panic!("Expected a changed result");
            }
        };
        println!("After parse_and_transform ({passes}):\n{}", result);
        assert!(result.contains("define i32 @main"));
    }
}
