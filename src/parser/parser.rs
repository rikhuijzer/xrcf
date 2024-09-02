use crate::dialect::arith;
use crate::dialect::func;
use crate::dialect::llvmir;
use crate::ir;
use crate::ir::operation::OperationName;
use crate::ir::Block;
use crate::ir::ModuleOp;
use crate::ir::Op;
use crate::ir::Operation;
use crate::ir::Region;
use crate::parser::scanner::Scanner;
use crate::parser::token::Token;
use crate::parser::token::TokenKind;
use anyhow::Result;
use std::any::Any;
use std::sync::Arc;

pub struct Parser<T: Parse> {
    src: String,
    tokens: Vec<Token>,
    current: usize,
    parse_op: std::marker::PhantomData<T>,
}

impl<T: Parse> Parser<T> {
    pub fn tokens(&self) -> &Vec<Token> {
        &self.tokens
    }
}

/// Downstream crates can implement this trait to support custom parsing.
/// The default implementation can only know about operations defined in
/// this crate.
/// This avoids having some global hashmap registry of all possible operations.
pub trait Parse {
    fn op<T: Parse>(parser: &mut Parser<T>) -> Result<Arc<dyn Op>>
    where
        Self: Sized;
}

/// Default operation parser.
/// This parser knows about all operations defined in this crate.
/// For operations in external dialects, define another parser and use it.
pub struct BuiltinParse;

enum Dialects {
    Builtin,
    LLVM,
}

impl Parse for BuiltinParse {
    fn op<T: Parse>(parser: &mut Parser<T>) -> Result<Arc<dyn Op>> {
        if parser.peek().lexeme == "return" {
            return <func::ReturnOp as Parse>::op(parser);
        }
        let name = if parser.peek().kind == TokenKind::Equal {
            // Ignore result name and '='.
            parser.peek_n(2)
        } else {
            parser.peek()
        };
        let name = name.lexeme.clone();
        match name.as_str() {
            "llvm.mlir.global" => <llvmir::op::GlobalOp as Parse>::op(parser),
            "func.func" => <ir::FuncOp as Parse>::op(parser),
            "arith.addi" => <arith::AddiOp as Parse>::op(parser),
            _ => Err(anyhow::anyhow!("Unknown operation: {}", name)),
        }
    }
}

impl<T: Parse> Parser<T> {
    pub fn previous(&self) -> &Token {
        &self.tokens[self.current - 1]
    }
    pub fn previous_n(&self, n: usize) -> &Token {
        &self.tokens[self.current - n]
    }
    pub fn advance(&mut self) -> &Token {
        if !self.is_at_end() {
            self.current += 1;
        }
        self.previous()
    }
    pub fn peek(&self) -> &Token {
        self.tokens.get(self.current).unwrap()
    }
    pub fn peek_n(&self, n: usize) -> &Token {
        self.tokens.get(self.current + n).unwrap()
    }
    fn is_at_end(&self) -> bool {
        self.peek().kind == TokenKind::Eof
    }
    pub fn check(&self, kind: TokenKind) -> bool {
        if self.is_at_end() {
            return false;
        }
        self.peek().kind == kind
    }
    pub fn report_error(&self, token: &Token, msg: &str) -> Result<Token> {
        let msg = Scanner::report_error(&self.src, &token.location, msg);
        Err(anyhow::anyhow!(format!("\n\n{msg}\n")))
    }
    pub fn report_token_error(&self, token: &Token, expected: TokenKind) -> Result<Token> {
        let msg = format!(
            "Expected {:?}, but got \"{}\" of kind {:?}",
            expected, token.lexeme, token.kind
        );
        let msg = Scanner::report_error(&self.src, &token.location, &msg);
        Err(anyhow::anyhow!(format!("\n\n{msg}\n")))
    }
    pub fn expect(&mut self, kind: TokenKind) -> Result<Token> {
        if self.check(kind) {
            self.advance();
            Ok(self.previous().clone())
        } else {
            self.report_token_error(self.peek(), kind)
        }
    }
    pub fn block(&mut self) -> Result<Block> {
        // Not all blocks have a label.
        // let label = self.expect(TokenKind::PercentIdentifier)?;
        // let label = label.lexeme.clone();
        // println!("label: {}", label);
        // let _equal = self.expect(TokenKind::Equal)?;
        let arguments = vec![];
        let mut ops = vec![];
        while let Ok(op) = T::op(self) {
            ops.push(op.clone());
            if op.is_terminator() {
                break;
            }
        }
        if ops.is_empty() {
            let token = self.peek();
            self.report_error(&token, "Could not find operations in block")?;
        }
        Ok(Block::new(None, arguments, ops))
    }
    pub fn match_kinds(&mut self, kinds: &[TokenKind]) -> bool {
        for kind in kinds {
            if self.check(*kind) {
                self.advance();
                return true;
            }
        }
        false
    }
    pub fn region(&mut self) -> Result<Region> {
        let _lbrace = self.expect(TokenKind::LBrace)?;
        let mut blocks = vec![];
        // while !self.check(TokenKind::RBrace) {
        let block = self.block()?;
        blocks.push(Box::pin(block));
        // }
        self.advance();
        Ok(Region::new(blocks))
    }
    pub fn parse(src: &str) -> Result<ModuleOp> {
        let mut parser = Parser::<T> {
            src: src.to_string(),
            tokens: Scanner::scan(src)?,
            current: 0,
            parse_op: std::marker::PhantomData,
        };
        let op: Arc<dyn Op> = T::op(&mut parser)?;
        let any_op: Box<dyn Any> = Box::new(op.clone());
        let op: ModuleOp = if let Ok(module_op) = any_op.downcast::<ModuleOp>() {
            *module_op
        } else {
            let name = OperationName::new("module".to_string());
            let ops: Vec<Arc<dyn Op>> = vec![op];
            let block = Block::new(None, vec![], ops);
            let region = Region::new(vec![Box::pin(block)]);
            let mut operation = Operation::default();
            operation.set_name(name).set_region(region);
            let module_op = ModuleOp::from_operation(Box::pin(operation));
            module_op.unwrap()
        };
        Ok(op)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Op;

    #[test]
    fn parse_global() {
        // From test/Target/LLVMIR/llvmir.mlir
        let src = "llvm.mlir.global internal @i32_global(42 : i32) : i32";
        let module_op = Parser::<BuiltinParse>::parse(src).unwrap();
        assert_eq!(module_op.operation().name(), "module");
        // let body = module_op.operation().get_body_region();
        // assert_eq!(body.blocks().len(), 1);

        let repr = format!("{:#}", module_op);
        let lines: Vec<&str> = repr.split('\n').collect();
        println!("{}", repr);
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], "module {");
        assert_eq!(lines[1], "  llvm.mlir.global internal @i32_global(42) ");
        assert_eq!(lines[2], "}");
    }
}
