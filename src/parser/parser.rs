use crate::parser::scanner::Scanner;
use crate::ir::operation::Operation;
use crate::parser::token::Token;
use anyhow::Result;
struct Parser {
    tokens: Vec<Token>,
    current: usize,
}

impl Parser {
    fn operation(&mut self) -> Result<Operation> {
        todo!()
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