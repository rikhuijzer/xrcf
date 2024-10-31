use crate::python;
use crate::python_to_mlir::ConvertPythonToMLIR;
use anyhow::Result;
use std::sync::Arc;
use std::sync::RwLock;
use xrcf::convert::Pass;
use xrcf::convert::RewriteResult;
use xrcf::ir::spaces;
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

fn is_function_call<T: ParserDispatch>(parser: &Parser<T>) -> bool {
    let syntax_matches = parser.peek().kind == TokenKind::BareIdentifier
        && parser.peek_n(1).unwrap().kind == TokenKind::LParen;
    let known_keyword = {
        if let TokenKind::BareIdentifier = parser.peek().kind {
            match parser.peek().lexeme.as_str() {
                "def" => true,
                "print" => true,
                _ => false,
            }
        } else {
            false
        }
    };
    syntax_matches && !known_keyword
}

impl ParserDispatch for PythonParserDispatch {
    fn parse_op(
        parser: &mut Parser<Self>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        if is_function_call(parser) {
            return <python::CallOp as Parse>::op(parser, parent);
        }
        let name = if parser.peek_n(1).unwrap().kind == TokenKind::Equal {
            // Ignore result name and '=' (e.g., `x = <op name>`).
            parser.peek_n(2).unwrap().clone()
        } else {
            // Ignore nothing (e.g., `<op name> x, y`).
            parser.peek().clone()
        };
        match name.lexeme.clone().as_str() {
            "def" => <python::FuncOp as Parse>::op(parser, parent),
            "print" => <python::PrintOp as Parse>::op(parser, parent),
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

/// Replace Python's indentation with brackets to make it easier to parse.
///
/// For example, this function changes
/// ```python
/// def main():
///     print("Hello, World!")
/// ```
/// to
/// ```python
/// def main() {
///     print("Hello, World!")
/// }
/// ```
fn replace_indentation(src: &str) -> String {
    let mut result = String::new();
    let mut indent = 0;
    for line in src.lines() {
        if line.ends_with(":") {
            indent += 1;
            let line = line.trim_end_matches(':');
            result.push_str(&format!("{} {{", line));
        } else if line.starts_with(&format!("{}", spaces(2 * indent))) {
            let line = line.trim();
            result.push_str(&format!("{}{}", spaces(indent), line));
        } else {
            indent -= 1;
            result.push_str(&format!("{}}}", spaces(indent)));
            result.push('\n');
            let line = line.trim();
            result.push_str(&format!("{}{}", spaces(indent), line));
        }
        result.push('\n');
    }
    result
}

pub fn parse_and_transform(src: &str, passes: &Passes) -> Result<RewriteResult> {
    let src = replace_indentation(src);
    let op = Parser::<PythonParserDispatch>::parse(&src)?;
    let result = transform::<PythonTransformDispatch>(op, passes)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use indoc::indoc;
    use tracing;
    use xrcf::init_subscriber;
    use xrcf::ir::GuardedOp;

    /// Initialize the subscriber for the tests.
    ///
    /// Cannot pass options, since the tests run concurrently.
    pub fn init_tracing() {
        let level = tracing::Level::INFO;
        match init_subscriber(level) {
            Ok(_) => (),
            Err(_e) => (),
        }
    }

    #[test]
    fn test_replace_indentation() {
        init_tracing();
        let src = indoc! {r#"
        def main():
            print("Hello, World!")

        def hello():
            print("Hello, World!")

        hello()
        "#}
        .trim();
        let expected = indoc! {r#"
        def main() {
          print("Hello, World!")
        }

        def hello() {
          print("Hello, World!")
        }

        hello()
        "#}
        .trim();
        let result = replace_indentation(src);
        tracing::info!("Before replace_indentation:\n{src}\n");
        tracing::info!("After replace_indentation:\n{result}\n");
        assert_eq!(result.trim(), expected.trim());
    }

    fn print_heading(msg: &str, src: &str, passes: &Passes) {
        tracing::info!("{msg} ({passes}):\n```\n{src}\n```\n");
    }
    fn test_transform(src: &str, passes: Vec<&str>) -> (Arc<RwLock<dyn Op>>, String) {
        let src = src.trim();
        init_tracing();
        let passes = Passes::from_vec(passes);
        print_heading("Before", src, &passes);
        let result = parse_and_transform(src, &passes).unwrap();
        let new_root_op = match result {
            RewriteResult::Changed(changed_op) => changed_op.0,
            RewriteResult::Unchanged => {
                panic!("Expected changes");
            }
        };
        let actual = new_root_op.try_read().unwrap().to_string();
        print_heading("After", &actual, &passes);
        (new_root_op, actual)
    }

    #[test]
    fn test_default_dispatch() {
        init_tracing();
        let src = indoc! {r#"
        func.func @main() -> i32 {
            %0 = arith.constant 0 : i32
            return %0 : i32
        }
        "#};
        let passes = vec!["--convert-func-to-llvm", "--convert-mlir-to-llvmir"];
        let (_module, actual) = test_transform(src, passes);
        assert!(actual.contains("define i32 @main"));
    }

    fn compare_lines(expected: &str, actual: &str) {
        let lines = expected.lines().zip(actual.lines());
        for (i, (expected_line, actual_line)) in lines.enumerate() {
            assert_eq!(expected_line.trim(), actual_line.trim(), "Line {i} differs");
        }
    }

    #[test]
    fn test_hello_world() {
        init_tracing();
        let src = indoc! {r#"
        def hello():
            print("Hello, World!")

        hello()
        "#}
        .trim();
        let expected = indoc! {r#"
        module {
          func.func @hello() {
            unstable.printf("Hello, World!")
            return
          }
          func.func @main() -> i32 {
            %0 = arith.constant 0 : i32
            func.call @hello() : () -> ()
            return %0 : i32
          }
        }
        "#}
        .trim();
        let passes = vec!["--convert-python-to-mlir"];
        let (_module, actual) = test_transform(src, passes);
        compare_lines(expected, actual.trim());

        let passes = vec!["--convert-python-to-mlir", "--convert-unstable-to-mlir"];
        let (module, _actual) = test_transform(src, passes);

        let ops = module.ops();
        println!("ops[1]: {}", ops[1].operation().try_read().unwrap());
        assert_eq!(ops.len(), 3);
        for (i, op) in ops.iter().enumerate() {
            let operation = op.operation();
            let operation = operation.try_read().unwrap();
            assert!(
                operation.parent().is_some(),
                "op {i}: {operation} has no parent"
            );
        }
    }
}
