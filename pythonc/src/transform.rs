use anyhow::Result;
use std::sync::Arc;
use std::sync::RwLock;
use xrcf::convert::RewriteResult;
use xrcf::ir::Op;
use xrcf::parser::DefaultParserDispatch;
use xrcf::parser::Parser;
use xrcf::transform;
use xrcf::DefaultTransformDispatch;
use xrcf::Passes;
use xrcf::TransformDispatch;

struct PythonTransformDispatch;

impl TransformDispatch for PythonTransformDispatch {
    fn dispatch(op: Arc<RwLock<dyn Op>>, passes: &Passes) -> Result<RewriteResult> {
        println!("here");
        let result = DefaultTransformDispatch::dispatch(op, passes)?;
        Ok(result)
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
        let passes = vec![
            "--convert-func-to-llvm".to_string(),
            "--convert-mlir-to-llvmir".to_string(),
        ];
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
        let passes = passes.vec().join(" ");
        println!("After parse_and_transform ({passes}):\n{}", result);
        assert!(result.contains("define i32 @main"));
    }
}
