use crate::dialect::llvmir;
use crate::ir::operation::Operation;
use crate::ir::operation::OperationName;
use crate::ir::Block;
use crate::ir::ModuleOp;
use crate::ir::Op;
use crate::ir::Region;
use crate::ir;
use crate::parser::scanner::Scanner;
use crate::parser::token::Token;
use crate::parser::token::TokenKind;
use anyhow::Result;
use std::any::Any;
use std::sync::Arc;

pub struct Parser<T: Parse> {
    tokens: Vec<Token>,
    current: usize,
    parse_op: std::marker::PhantomData<T>,
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

struct DefaultParse;

enum Dialects {
    Builtin,
    LLVM,
}

impl Parse for DefaultParse {
    /// Default operation parser.
    fn op<T: Parse>(parser: &mut Parser<T>) -> Result<Arc<dyn Op>> {
        let name = parser.advance();
        let name = OperationName::new(name.lexeme.clone());
        match name.name().as_str() {
            "llvm.mlir.global" => <llvmir::op::GlobalOp as Parse>::op(parser),
            "func.func" => <ir::FuncOp as Parse>::op(parser),
            _ => Err(anyhow::anyhow!("Unknown operation: {}", name.name())),
        }
    }
}

impl<T: Parse> Parser<T> {
    pub fn previous(&self) -> &Token {
        &self.tokens[self.current - 1]
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
    fn is_at_end(&self) -> bool {
        self.peek().kind == TokenKind::Eof
    }
    fn block(&mut self) -> Result<Block> {
        let label = self.advance().lexeme.clone();
        let arguments = vec![];
        let ops = vec![T::op(self)?];
        Ok(Block::new(label, arguments, ops))
    }
    pub fn check(&self, kind: TokenKind) -> bool {
        if self.is_at_end() {
            return false;
        }
        self.peek().kind == kind
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
    fn region(&mut self) -> Result<Region> {
        if !self.check(TokenKind::LBrace) {
            todo!("Expected region to start with a '{{'");
        }
        self.advance();
        let mut blocks = vec![];
        while !self.check(TokenKind::RBrace) {
            let block = self.block()?;
            blocks.push(Box::pin(block));
        }
        self.advance();
        Ok(Region::new(blocks))
    }
    fn regions(&mut self) -> Result<Vec<Region>> {
        let mut regions = vec![];
        while self.check(TokenKind::LBrace) {
            let region = self.region()?;
            regions.push(region);
        }
        Ok(regions)
    }
    pub fn parse(src: &str) -> Result<ModuleOp> {
        let mut parser = Parser::<T> {
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
            let attributes = vec![];
            let ops: Vec<Arc<dyn Op>> = vec![op];
            let block = Block::new("".to_string(), vec![], ops);
            let region = Region::new(vec![Box::pin(block)]);
            let regions = vec![Box::pin(region)];
            let operation = Operation::new(name, attributes, regions, None);
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
    fn parse_func() {
        let src = "
          func.func @test_addi(%arg0 : i64, %arg1 : i64) -> i64 {
            %0 = arith.addi %arg0, %arg1 : i64
            return %0 : i64
          }
        ";
        let module_op = Parser::<DefaultParse>::parse(src).unwrap();
        assert_eq!(module_op.operation().name(), "module");
    }

    #[test]
    fn parse_global() {
        // From test/Target/LLVMIR/llvmir.mlir
        let src = "llvm.mlir.global internal @i32_global(42 : i32) : i32";
        let module_op = Parser::<DefaultParse>::parse(src).unwrap();
        assert_eq!(module_op.operation().name(), "module");
        // let body = module_op.operation().get_body_region();
        // assert_eq!(body.blocks().len(), 1);

        let repr = format!("{:#}", module_op);
        let lines: Vec<&str> = repr.split('\n').collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], "module {");
        assert_eq!(lines[1], "  llvm.mlir.global internal @i32_global(42) ");
        assert_eq!(lines[2], "}");
    }
}
