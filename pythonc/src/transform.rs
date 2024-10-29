use crate::python_to_mlir::ConvertPythonToMLIR;
use anyhow::Result;
use std::sync::Arc;
use std::sync::RwLock;
use xrcf::convert::Pass;
use xrcf::convert::RewriteResult;
use xrcf::ir::Op;
use xrcf::parser::DefaultParserDispatch;
use xrcf::parser::Parser;
use xrcf::transform;
use xrcf::DefaultTransformDispatch;
use xrcf::Passes;
use xrcf::SinglePass;
use xrcf::TransformDispatch;

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
    let op = Parser::<DefaultParserDispatch>::parse(src)?;
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
