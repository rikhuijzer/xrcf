//! Library for the Wea compiler.
//!
//! This library is called from the cli inside `main.rs` and also from the tests
//! inside the tests directory.
//!
//! To import some `X` from inside this library, it may sometimes be necessary
//! to add a `pub use::X` to `main.rs`.
#![allow(clippy::new_without_default)]
#![allow(clippy::arc_with_non_send_sync)]

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
use xrcf::ir::BlockArgument;
use xrcf::ir::BlockArgumentName;
use xrcf::ir::Op;
use xrcf::ir::Operation;
use xrcf::ir::Type;
use xrcf::ir::Value;
use xrcf::ir::Values;
use xrcf::shared::Shared;
use xrcf::DefaultTransformDispatch;
use xrcf::SinglePass;
use xrcf::TransformDispatch;

pub trait WeaParse {
    fn is_wea_function_argument(&mut self) -> bool;
    fn parse_wea_function_argument(&mut self) -> Result<Shared<Value>>;
    fn parse_wea_function_arguments_into(&mut self, operation: &mut Operation) -> Result<()>;
    fn defines_result(&mut self) -> bool;
}

impl<T: ParserDispatch> WeaParse for Parser<T> {
    fn is_wea_function_argument(&mut self) -> bool {
        // For example, `a` in `fn plus(a: i32)`.
        self.check(TokenKind::BareIdentifier)
    }
    /// Parse typed parameter like `a: i32`.
    fn parse_wea_function_argument(&mut self) -> Result<Shared<Value>> {
        let identifier = self.expect(TokenKind::BareIdentifier)?;
        let name = identifier.lexeme.clone();
        self.expect(TokenKind::Colon)?;
        let typ = T::parse_type(self)?;
        let name = BlockArgumentName::Name(name);
        let arg = Value::BlockArgument(BlockArgument::new(name, typ));
        let operand = Shared::new(arg.into());
        if self.check(TokenKind::Comma) {
            self.advance();
        }
        Ok(operand)
    }
    /// Parse typed parameters like `a: i32` into operation.
    fn parse_wea_function_arguments_into(&mut self, operation: &mut Operation) -> Result<()> {
        self.expect(TokenKind::LParen)?;
        let mut operands = vec![];
        while self.is_wea_function_argument() {
            operands.push(self.parse_wea_function_argument()?);
        }
        self.expect(TokenKind::RParen)?;
        let values = Values::from_vec(operands);
        operation.set_arguments(values.clone());
        Ok(())
    }
    /// Returns whether the op defines a result.
    ///
    /// This method is used on ops that are expected to define a result. If they
    /// don't then that is assumed to mean that the op is used as a return.
    fn defines_result(&mut self) -> bool {
        self.peek_n(1).unwrap().kind == TokenKind::Equal
    }
}

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
            indent_level += 1;
        } else {
            result.push_str(trimmed);
        }

        result.push('\n');
    }

    while indent_level > 0 {
        indent_level -= 1;
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
        if parent.is_none() {
            panic!("only modules and functions can have no parent");
        }
        let second = parser.peek_n(1).unwrap();
        let op = match second.lexeme.as_str() {
            // Plus/arith should probably become a tree-like structure. During
            // conversion to MLIR, this can be flattened to the MLIR ssa
            // representation.
            //
            // So probably need to find the last operation that is executed and
            // then during the parsing of that, descend into the arith. Is
            // similar to func func where the parser also descends into the
            // body.
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
