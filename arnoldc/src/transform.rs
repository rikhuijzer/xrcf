use crate::arnold;
use crate::arnold_to_mlir::ConvertArnoldToMLIR;
use anyhow::Result;
use xrcf::convert::Pass;
use xrcf::convert::RewriteResult;
use xrcf::frontend::default_dispatch;
use xrcf::frontend::default_parse_type;
use xrcf::frontend::Parse;
use xrcf::frontend::Parser;
use xrcf::frontend::ParserDispatch;
use xrcf::frontend::TokenKind;
use xrcf::ir::Block;
use xrcf::ir::Op;
use xrcf::ir::Type;
use xrcf::shared::Shared;
use xrcf::transform;
use xrcf::DefaultTransformDispatch;
use xrcf::SinglePass;
use xrcf::TransformDispatch;
use xrcf::TransformOptions;
pub struct ArnoldParserDispatch;

fn is_function_call<T: ParserDispatch>(parser: &Parser<T>) -> bool {
    let syntax_matches = parser.peek().kind == TokenKind::BareIdentifier
        && parser.peek_n(1).unwrap().kind == TokenKind::LParen;
    let known_keyword = {
        if let TokenKind::BareIdentifier = parser.peek().kind {
            matches!(parser.peek().lexeme.as_str(), "def" | "print")
        } else {
            false
        }
    };
    syntax_matches && !known_keyword
}

impl ParserDispatch for ArnoldParserDispatch {
    fn parse_op(
        parser: &mut Parser<Self>,
        parent: Option<Shared<Block>>,
    ) -> Result<Shared<dyn Op>> {
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

        let name = default_parse_name(parser);
        default_dispatch(name, parser, parent)
    }
    fn parse_type(parser: &mut Parser<Self>) -> Result<Shared<dyn Type>> {
        default_parse_type(parser)
    }
}

pub struct ArnoldTransformDispatch;

impl TransformDispatch for ArnoldTransformDispatch {
    fn dispatch(op: Shared<dyn Op>, pass: &SinglePass) -> Result<RewriteResult> {
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
/// logic separated (i.e., to make it easier to understand the code).
fn preprocess(src: &str) -> String {
    let mut result = String::new();
    for line in src.lines() {
        #[allow(clippy::if_same_then_else)]
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
            result.push_str(line);
        }
        result.push('\n');
    }
    result
}

pub fn parse_and_transform(src: &str, options: &TransformOptions) -> Result<RewriteResult> {
    let src = preprocess(src);
    let op = Parser::<ArnoldParserDispatch>::parse(&src)?;
    let result = transform::<ArnoldTransformDispatch>(op, options)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile_passes;
    use indoc::indoc;
    use std::panic::Location;
    use xrcf::shared::SharedExt;
    use xrcf::tester::Tester;
    use xrcf::Passes;

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

    fn test_transform(src: &str, passes: Vec<&str>) -> (Shared<dyn Op>, String) {
        Tester::init_tracing();
        let src = src.trim();
        let passes = Passes::from_vec(passes);
        print_heading("Before", src, &passes);
        let options = TransformOptions::from_passes(passes.clone());
        let result = parse_and_transform(src, &options).unwrap();
        let new_root_op = match result {
            RewriteResult::Changed(changed_op) => changed_op.op,
            RewriteResult::Unchanged => {
                panic!("Expected changes");
            }
        };
        let actual = new_root_op.rd().to_string();
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
            %0 = arith.constant 1 : i1
            experimental.printf("x: ")
            experimental.printf("%d", %0)
            %1 = arith.constant 0 : i32
            return %1 : i32
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
          %0 = arith.constant 0 : i1
          scf.if %0 {
            experimental.printf("x was true")
          } else {
            experimental.printf("x was false")
          }
          %1 = arith.constant 0 : i32
          return %1 : i32
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
