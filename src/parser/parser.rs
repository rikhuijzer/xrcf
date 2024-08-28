use crate::parser::scanner::Scanner;
use crate::ir::operation::Operation;
use crate::ir::operation::OperationName;
use crate::parser::token::Token;
use anyhow::Result;
use crate::parser::token::TokenKind;
use crate::ir::Region;
use crate::ir::Block;
use crate::ir::Op;
use std::pin::pin;
use std::pin::Pin;

struct Parser {
    tokens: Vec<Token>,
    current: usize,
}

impl Parser {
    fn previous(&self) -> &Token {
        &self.tokens[self.current - 1]
    }
    fn advance(&mut self) -> &Token {
        if !self.is_at_end() {
            self.current += 1;
        }
        self.previous()
    }
    fn peek(&self) -> &Token {
        self.tokens.get(self.current).unwrap()
    }
    fn is_at_end(&self) -> bool {
        self.peek().kind == TokenKind::Eof
    }
    fn block(&mut self) -> Result<Block> {
        let label = self.advance().lexeme.clone();
        let arguments = vec![];
        let operations = vec![self.operation()?];
        Ok(Block::new(label, arguments, operations))
    }
    fn check(&self, kind: TokenKind) -> bool {
        if self.is_at_end() {
            return false;
        }
        self.peek().kind == kind
    }
    fn match_kinds(&mut self, kinds: &[TokenKind]) -> bool {
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
            blocks.push(block);
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
    fn operation(&mut self) -> Result<Operation> {
        let name = self.advance();
        let name = OperationName::new(name.lexeme.clone());
        let regions = self.regions()?;
        let operation = Operation::new(name, regions, None);
        Ok(operation)
    }
    pub fn parse(src: &str) -> Result<Operation> {
        let mut parser = Parser {
            tokens: Scanner::scan(src)?,
            current: 0,
        };
        let operation = parser.operation()?;
        let operation = if operation.name() != "module" {
            let name = OperationName::new("module".to_string());
            let block = Block::new("entry".to_string(), vec![], vec![operation]);
            let region = Region::new(vec![block]);
            Operation::new(name, vec![region], None)
        } else {
            operation
        };
        Ok(operation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::ModuleOp;

    #[test]
    fn test_parse() {
        let src = "llvm.mlir.global internal @i32_global(42 : i32) : i32";
        let operation = Parser::parse(src).unwrap();
        assert_eq!(operation.name(), "module");
        let module_op = ModuleOp::from_operation(operation.clone()).unwrap();
        let body = module_op.getBodyRegion();
        assert_eq!(body.blocks().len(), 1);
        println!("{}", operation);
        // assert!(false);
    }
}