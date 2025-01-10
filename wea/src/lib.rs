mod op;
mod wea_to_mlir;

use anyhow::Result;
use wea_to_mlir::ConvertWeaToMLIR;
use xrcf::convert::Pass;
use xrcf::convert::RewriteResult;
use xrcf::frontend::default_dispatch;
use xrcf::frontend::default_parse_name;
use xrcf::frontend::default_parse_type;
use xrcf::frontend::Parse;
use xrcf::frontend::Parser;
use xrcf::frontend::ParserDispatch;
use xrcf::frontend::Token;
use xrcf::frontend::TokenKind;
use xrcf::ir::Block;
use xrcf::ir::Op;
use xrcf::ir::Type;
use xrcf::shared::Shared;
use xrcf::DefaultTransformDispatch;
use xrcf::SinglePass;
use xrcf::TransformDispatch;

pub struct WeaParserDispatch;

fn indentation_to_braces(src: &str) -> String {
    let mut result = String::new();
    let mut indent_level = 0;

    for line in src.lines() {
        let trimmed = line.trim();

        result.push_str(&xrcf::ir::spaces(indent_level));

        if let Some(stripped) = trimmed.strip_suffix(':') {
            result.push_str(stripped);
            result.push_str(" {");
            indent_level += 2;
        } else {
            result.push_str(trimmed);
        }

        result.push('\n');
    }

    while indent_level > 0 {
        indent_level -= 2;
        result.push_str(&xrcf::ir::spaces(indent_level));
        result.push_str("}\n");
    }

    result
}

fn is_function_definition<T: ParserDispatch>(parser: &Parser<T>) -> bool {
    fn is_func(token: &Token) -> bool {
        token.kind == TokenKind::BareIdentifier && token.lexeme == "fn"
    }
    is_func(parser.peek()) || is_func(parser.peek_n(1).unwrap())
}

impl ParserDispatch for WeaParserDispatch {
    /// Preprocess Wea source code to make it easier to parse.
    ///
    /// This mostly adds braces to make the structure of the code more clear. This
    /// could have been done in the xrcf parser, but it's moved here to keep the
    /// logic separated (i.e., to make it easier to understand the code).
    fn preprocess(src: &str) -> String {
        indentation_to_braces(src)
    }
    fn parse_op(
        parser: &mut Parser<Self>,
        parent: Option<Shared<Block>>,
    ) -> Result<Shared<dyn Op>> {
        if is_function_definition(parser) {
            return <op::FuncOp as Parse>::op(parser, parent);
        }
        let first = parser.peek();
        let second = parser.peek_n(1).unwrap();
        let first_two = format!("{} {}", first.lexeme, second.lexeme);
        let op = match first_two.as_str() {
            "+" => Some(<op::PlusOp as Parse>::op(parser, parent.clone())),
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

pub struct WeaTransformDispatch;

impl TransformDispatch for WeaTransformDispatch {
    fn dispatch(op: Shared<dyn Op>, pass: &SinglePass) -> Result<RewriteResult> {
        match pass.to_string().as_str() {
            ConvertWeaToMLIR::NAME => ConvertWeaToMLIR::convert(op),
            _ => DefaultTransformDispatch::dispatch(op, pass),
        }
    }
}
