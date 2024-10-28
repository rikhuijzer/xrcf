use anyhow::Result;
use std::sync::Arc;
use std::sync::RwLock;
use xrcf::convert::RewriteResult;
use xrcf::ir::Op;
use xrcf::parser::DefaultParserDispatch;
use xrcf::parser::Parser;
use xrcf::transform;
use xrcf::DefaultTransformDispatch;
use xrcf::TransformDispatch;
extern crate xrcf;

struct PythonTransformDispatch;

impl TransformDispatch for PythonTransformDispatch {
    fn dispatch(op: Arc<RwLock<dyn Op>>, passes: Vec<&str>) -> Result<RewriteResult> {
        let result = DefaultTransformDispatch::dispatch(op, passes)?;
        Ok(result)
    }
}

fn parse_and_transform(src: &str, arguments: &str) -> Result<RewriteResult> {
    let op = Parser::<DefaultParserDispatch>::parse(src)?;
    let result = transform::<PythonTransformDispatch>(op, arguments)?;
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
            unstable.printf("hello, world\n")
            return %0 : i32
        }
        "#};
        let arguments = "--convert-unstable-to-mlir";
        let result = parse_and_transform(src, arguments).unwrap();
        let result = match result {
            RewriteResult::Changed(op) => {
                let op = op.0.try_read().unwrap();
                op.to_string()
            }
            RewriteResult::Unchanged => {
                panic!("Expected a changed result");
            }
        };
        println!("After parse_and_transform ({arguments}):\n{}", result);
        assert!(false);
    }
}
