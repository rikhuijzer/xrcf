use crate::parser::scanner::Scanner;
use crate::ir::operation::Operation;
use crate::ir::operation::OperationName;
use crate::parser::token::Token;
use anyhow::Result;
use crate::parser::token::TokenKind;
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
    fn operation(&mut self) -> Result<Operation> {
        let name = self.advance();
        let name = OperationName::new(name.lexeme.clone());
        let operation = Operation::new(name);
        Ok(operation)
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
    pub fn parse(src: &str) -> Result<Operation> {
        let mut parser = Parser {
            tokens: Scanner::scan(src)?,
            current: 0,
        };
        let operation = if true { // not module
            // create module with a nested operation below it
            let name = OperationName::new("module".to_string());
            Ok(Operation::new(name))
        } else {
            parser.operation()
        };
        operation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_parse() {
        let src = "llvm.mlir.global internal @i32_global(42 : i32) : i32";
        let operation = Parser::parse(src).unwrap();
        assert_eq!(operation.name(), "module");
    }
}