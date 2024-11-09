use crate::arnold;
use crate::arnold_to_mlir::ConvertArnoldToMLIR;
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

struct ExampleParserDispatch;

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

impl ParserDispatch for ExampleParserDispatch {
    fn parse_op(
        parser: &mut Parser<Self>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        if is_function_call(parser) {
            return <arnold::CallOp as Parse>::op(parser, parent);
        }
        let first = parser.peek();
        let second = parser.peek_n(1).unwrap();
        let first_two = format!("{} {}", first.lexeme, second.lexeme);
        if first_two == "TALK TO" {
            return <arnold::PrintOp as Parse>::op(parser, parent);
        }
        let name = if parser.peek_n(1).unwrap().kind == TokenKind::Equal {
            // Ignore result name and '=' (e.g., `x = <op name>`).
            parser.peek_n(2).unwrap().clone()
        } else {
            // Ignore nothing (e.g., `<op name> x, y`).
            parser.peek().clone()
        };
        match name.lexeme.clone().as_str() {
            "def" => <arnold::FuncOp as Parse>::op(parser, parent),
            "print" => <arnold::PrintOp as Parse>::op(parser, parent),
            _ => default_dispatch(name, parser, parent),
        }
    }
    fn parse_type(_parser: &mut Parser<Self>) -> Result<Arc<RwLock<dyn Type>>> {
        todo!()
    }
}

struct ExampleTransformDispatch;

impl TransformDispatch for ExampleTransformDispatch {
    fn dispatch(op: Arc<RwLock<dyn Op>>, pass: &SinglePass) -> Result<RewriteResult> {
        match pass.to_string().as_str() {
            ConvertArnoldToMLIR::NAME => ConvertArnoldToMLIR::convert(op),
            _ => DefaultTransformDispatch::dispatch(op, pass),
        }
    }
}

/// Replace begin and end of blocks with braces to make it easier to parse.
///
/// Also adds some indentation for readability.
/// For example, this function changes
/// ```arnoldc
/// IT'S SHOWTIME
/// TALK TO THE HAND "Hello, World!"
/// YOU HAVE BEEN TERMINATED
/// ```
/// to
/// ```mlir
/// def @main() {
///   TALK TO THE HAND "Hello, World!"
/// }
/// ```
fn replace_begin_and_end(src: &str) -> String {
    let mut result = String::new();
    let mut indent = 0;
    for line in src.lines() {
        if line.contains("IT'S SHOWTIME") {
            indent += 1;
            result.push_str(&format!("{}def main() {{", spaces(indent)));
        } else if line.contains("YOU HAVE BEEN TERMINATED") {
            indent -= 1;
            result.push_str(&format!("{}}}", spaces(indent)));
        } else {
            let line = line.trim();
            result.push_str(&format!("{}{}", spaces(indent), line));
        }
        result.push('\n');
    }
    result
}

pub fn parse_and_transform(src: &str, passes: &Passes) -> Result<RewriteResult> {
    let src = replace_begin_and_end(src);
    let op = Parser::<ExampleParserDispatch>::parse(&src)?;
    let result = transform::<ExampleTransformDispatch>(op, passes)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use indoc::indoc;
    use std::panic::Location;
    use tracing;
    use xrcf::tester::Tester;

    #[test]
    fn test_replace_begin_and_end() {
        Tester::init_tracing();
        let src = indoc! {r#"
        IT'S SHOWTIME
        TALK TO THE HAND "Hello, World!\n"
        YOU HAVE BEEN TERMINATED
        "#}
        .trim();
        let expected = indoc! {r#"
        def main() {
          TALK TO THE HAND "Hello, World!\n"
        }
        "#}
        .trim();
        let result = replace_begin_and_end(src);
        tracing::info!("Before replace_begin_and_end:\n{src}\n");
        tracing::info!("After replace_begin_and_end:\n{result}\n");
        assert_eq!(result.trim(), expected.trim());
    }

    fn print_heading(msg: &str, src: &str, passes: &Passes) {
        tracing::info!("{msg} ({passes}):\n```\n{src}\n```\n");
    }
    fn test_transform(src: &str, passes: Vec<&str>) -> (Arc<RwLock<dyn Op>>, String) {
        Tester::init_tracing();
        let src = src.trim();
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
        Tester::init_tracing();
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

    #[test]
    fn test_hello_world() {
        Tester::init_tracing();
        let src = indoc! {r#"
        IT'S SHOWTIME
        TALK TO THE HAND "Hello, World!\n"
        YOU HAVE BEEN TERMINATED
        "#}
        .trim();
        let expected = indoc! {r#"
        module {
          func.func @main() -> i32 {
            experimental.printf("Hello, World!\0A")
            %0 = arith.constant 0 : i32
            return %0 : i32
          }
        }
        "#}
        .trim();
        let passes = vec!["--convert-arnold-to-mlir"];
        let (module, actual) = test_transform(src, passes);
        Tester::check_lines_exact(expected, actual.trim(), Location::caller());
        Tester::verify(module);

        let passes = vec![
            "--convert-arnold-to-mlir",
            "--convert-experimental-to-mlir",
            "--convert-func-to-llvm",
            "--convert-mlir-to-llvmir",
        ];
        let (module, actual) = test_transform(src, passes);
        Tester::verify(module);
        assert!(actual.contains("declare i32 @printf(ptr)"));
        assert!(actual.contains("define i32 @main()"));
    }
}
