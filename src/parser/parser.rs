use crate::parser::scanner::Scanner;
use crate::ir::operation::Operation;
use crate::ir::operation::OperationName;
use crate::parser::token::Token;
use anyhow::Result;
struct Parser {
    tokens: Vec<Token>,
    current: usize,
}

impl Parser {
    fn previous(&self) -> Token {
        self.tokens[self.current - 1].clone()
    }
    fn advance(&mut self) -> Token {
        self.current += 1;
        self.previous()
    }
    fn operation(&mut self) -> Result<Operation> {
        let name = self.advance();
        let name = OperationName::new(name.lexeme);
        let operation = Operation::new(name);
        Ok(operation)
    }
    pub fn parse(src: &str) -> Result<Operation> {
        let mut parser = Parser {
            tokens: Scanner::scan(src)?,
            current: 0,
        };
        let operation = parser.operation()?;
        Ok(operation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_parse() {
        let src = "llvm.mlir.global internal @i32_global(42 : i32) : i32";
        let operation = Parser::parse(src).unwrap();
        assert_eq!(operation.name(), "llvm.mlir.global");
    }
}