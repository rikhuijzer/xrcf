use crate::arnold;
use crate::arnold_to_mlir::ConvertArnoldToMLIR;
use anyhow::Result;
use std::sync::Arc;
use std::sync::RwLock;
use xrcf::convert::Pass;
use xrcf::convert::RewriteResult;
use xrcf::ir::Block;
use xrcf::ir::Op;
use xrcf::ir::Type;
use xrcf::parser::default_dispatch;
use xrcf::parser::default_parse_type;
use xrcf::parser::Parse;
use xrcf::parser::Parser;
use xrcf::parser::ParserDispatch;
use xrcf::parser::TokenKind;
use xrcf::transform;
use xrcf::DefaultTransformDispatch;
use xrcf::Passes;
use xrcf::SinglePass;
use xrcf::TransformDispatch;

pub struct ArnoldParserDispatch;

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

impl ParserDispatch for ArnoldParserDispatch {
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
        let op = match first_two.as_str() {
            "BECAUSE I" => Some(<arnold::IfOp as Parse>::op(parser, parent.clone())),
            "HEY CHRISTMAS" => Some(<arnold::DeclareIntOp as Parse>::op(parser, parent.clone())),
            "IT '" => Some(<arnold::BeginMainOp as Parse>::op(parser, parent.clone())),
            "TALK TO" => Some(<arnold::PrintOp as Parse>::op(parser, parent.clone())),
            "YOU SET" => Some(<arnold::SetInitialValueOp as Parse>::op(
                parser,
                parent.clone(),
            )),
            _ => None,
        };
        if let Some(op) = op {
            return op;
        }

        // If the syntax doesn't look like ArnoldC, fallback to the default
        // parser that can parse MLIR syntax.
        let name = if parser.peek_n(1).unwrap().kind == TokenKind::Equal {
            // Ignore result name and '=' (e.g., `x = <op name>`).
            match parser.peek_n(2) {
                Some(name) => name.clone(),
                None => panic!("Couldn't peek 2 tokens at {}", parser.peek()),
            }
        } else {
            // Ignore nothing (e.g., `<op name> x, y`).
            parser.peek().clone()
        };
        match name.lexeme.clone().as_str() {
            _ => default_dispatch(name, parser, parent),
        }
    }
    fn parse_type(parser: &mut Parser<Self>) -> Result<Arc<RwLock<dyn Type>>> {
        default_parse_type(parser)
    }
}

pub struct ArnoldTransformDispatch;

impl TransformDispatch for ArnoldTransformDispatch {
    fn dispatch(op: Arc<RwLock<dyn Op>>, pass: &SinglePass) -> Result<RewriteResult> {
        match pass.to_string().as_str() {
            ConvertArnoldToMLIR::NAME => ConvertArnoldToMLIR::convert(op),
            _ => DefaultTransformDispatch::dispatch(op, pass),
        }
    }
}

/// Preprocess ArnoldC source code to make it easier to parse.
///
/// This mostly adds braces to make the structure of the code more clear. This
/// could have been done in the xrcf parser, but it's moved here to keep the
/// logic more separate.
fn preprocess(src: &str) -> String {
    let mut result = String::new();
    for line in src.lines() {
        if line.contains("IT'S SHOWTIME") {
            result.push_str(&format!("{} {{", line));
        } else if line.contains("BECAUSE I'M GOING TO SAY PLEASE") {
            result.push_str(&format!("{} {{", line));
        } else if line.contains("YOU HAVE BEEN TERMINATED") {
            result.push_str(&line.replace("YOU HAVE BEEN TERMINATED", "}"));
        } else if line.contains("YOU HAVE NO RESPECT FOR LOGIC") {
            result.push_str(&line.replace("YOU HAVE NO RESPECT FOR LOGIC", "}"));
        } else if line.contains("BULLSHIT") {
            result.push_str(&line.replace("BULLSHIT", "} BULLSHIT {"));
        } else {
            result.push_str(&line);
        }
        result.push('\n');
    }
    result
}

pub fn parse_and_transform(src: &str, passes: &Passes) -> Result<RewriteResult> {
    let src = preprocess(src);
    let op = Parser::<ArnoldParserDispatch>::parse(&src)?;
    let result = transform::<ArnoldTransformDispatch>(op, passes)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile_passes;
    use indoc::indoc;
    use std::panic::Location;
    use tracing;
    use xrcf::tester::Tester;

    fn flags() -> Vec<&'static str> {
        vec!["--convert-arnold-to-mlir"]
    }

    #[test]
    fn test_preprocess() {
        Tester::init_tracing();
        let src = indoc! {r#"
        IT'S SHOWTIME
        TALK TO THE HAND "Hello, World!\n"

        HEY CHRISTMAS TREE x
        YOU SET US UP @I LIED

        BECAUSE I'M GOING TO SAY PLEASE x
          BECAUSE I'M GOING TO SAY PLEASE x
            TALK TO THE HAND "x was true"
          BULLSHIT
            TALK TO THE HAND "this case will never happen"
          YOU HAVE NO RESPECT FOR LOGIC
        BULLSHIT
          TALK TO THE HAND "x was false"
        YOU HAVE NO RESPECT FOR LOGIC

        YOU HAVE BEEN TERMINATED
        "#}
        .trim();
        let expected = indoc! {r#"
        IT'S SHOWTIME {
        TALK TO THE HAND "Hello, World!\n"

        HEY CHRISTMAS TREE x
        YOU SET US UP @I LIED

        BECAUSE I'M GOING TO SAY PLEASE x {
          BECAUSE I'M GOING TO SAY PLEASE x {
            TALK TO THE HAND "x was true"
          } BULLSHIT {
            TALK TO THE HAND "this case will never happen"
          }
        } BULLSHIT {
          TALK TO THE HAND "x was false"
        }

        }
        "#}
        .trim();
        let result = preprocess(src);
        tracing::info!("Before preprocess:\n{src}\n");
        tracing::info!("After preprocess:\n{result}\n");
        Tester::check_lines_exact(expected, result.trim(), Location::caller());
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
            RewriteResult::Changed(changed_op) => changed_op.op,
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
        let (module, actual) = test_transform(src, flags());
        Tester::check_lines_exact(expected, actual.trim(), Location::caller());
        Tester::verify(module);

        let passes = compile_passes();
        let (module, actual) = test_transform(src, passes);
        Tester::verify(module);
        assert!(actual.contains("declare i32 @printf(ptr)"));
        assert!(actual.contains("define i32 @main()"));
    }

    #[test]
    fn test_print_digit() {
        Tester::init_tracing();
        let src = indoc! {r#"
        IT'S SHOWTIME

        HEY CHRISTMAS TREE x
        YOU SET US UP @NO PROBLEMO

        TALK TO THE HAND "x: "
        TALK TO THE HAND x

        YOU HAVE BEEN TERMINATED
        "#}
        .trim();
        let expected = indoc! {r#"
        module {
          func.func @main() -> i32 {
            %x = arith.constant 1 : i16
            experimental.printf("x: ")
            experimental.printf("%d", %x)
            %0 = arith.constant 0 : i32
            return %0 : i32
          }
        }
        "#}
        .trim();
        let (module, actual) = test_transform(src, flags());
        Tester::check_lines_exact(expected, actual.trim(), Location::caller());
        Tester::verify(module);
    }

    #[test]
    fn test_if_else() {
        Tester::init_tracing();
        let src = indoc! {r#"
        IT'S SHOWTIME

        HEY CHRISTMAS TREE x
        YOU SET US UP @I LIED

        BECAUSE I'M GOING TO SAY PLEASE x
          TALK TO THE HAND "x was true"
        BULLSHIT
          TALK TO THE HAND "x was false"
        YOU HAVE NO RESPECT FOR LOGIC

        YOU HAVE BEEN TERMINATED
        "#}
        .trim();
        let expected = indoc! {r#"
        func.func @main() -> i32 {
          %x = arith.constant 0 : i16
          scf.if %x {
            experimental.printf("x was true")
          } else {
            experimental.printf("x was false")
          }
          %0 = arith.constant 0 : i32
          return %0 : i32
        }
        "#}
        .trim();
        let (module, actual) = test_transform(src, flags());
        Tester::verify(module);
        Tester::check_lines_contain(actual.trim(), expected, Location::caller());

        let expected = indoc! {r#"
        define i32 @main() {
        
        br i1 0, label %bb1, label %bb2

        ret i32 0
        "#}
        .trim();
        let passes = compile_passes();
        let (module, actual) = test_transform(src, passes);
        Tester::verify(module);
        Tester::check_lines_contain(actual.trim(), expected, Location::caller());
    }
}
