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

pub struct WeaParserDispatch;

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

impl ParserDispatch for WeaParserDispatch {
    /// Preprocess Wea source code to make it easier to parse.
    ///
    /// This mostly adds braces to make the structure of the code more clear. This
    /// could have been done in the xrcf parser, but it's moved here to keep the
    /// logic separated (i.e., to make it easier to understand the code).
    fn preprocess(_src: &str) -> String {
        todo!()
    }
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
            ConvertArnoldToMLIR::NAME => ConvertArnoldToMLIR::convert(op),
            _ => DefaultTransformDispatch::dispatch(op, pass),
        }
    }
}
