use std::fmt::Error;

mod token;

use token::Token;
use token::TokenKind;

struct Scanner {
    source: String,
    tokens: Vec<Token>,
    start: usize,
    current: usize,
    line: usize,
}

impl Scanner {
    fn new(source: String) -> Self {
        Scanner {
            source,
            tokens: Vec::new(),
            start: 0,
            current: 0,
            line: 1,
        }
    }
    fn is_at_end(&self) -> bool {
        self.current >= self.source.len()
    }
    fn advance(&mut self) -> char {
        let c = self.source.chars().nth(self.current).unwrap();
        self.current += 1;
        c
    }
    fn peek(&self) -> char {
        if self.is_at_end() {
            return '\0';
        }
        self.source.chars().nth(self.current).unwrap()
    }
    fn peek_next(&self) -> char {
        if self.current + 1 >= self.source.len() {
            return '\0';
        }
        self.source.chars().nth(self.current + 1).unwrap()
    }
    fn add_token(&mut self, kind: TokenKind) {
        let lexeme = self.source[self.start..self.current].to_string();
        self.tokens.push(Token {
            kind,
            lexeme,
            literal: None,
        });
    }
    fn match_next(&mut self, expected: char) -> bool {
        if self.is_at_end() {
            return false;
        }
        if self.peek() != expected {
            return false;
        }
        self.current += 1;
        true
    }
    fn number(&mut self) -> Result<(), Error> {
        while self.peek().is_digit(10) {
            self.advance();
        }
        let mut is_float = false;
        if self.peek() == '.' && self.peek_next().is_digit(10) {
            if self.peek() == '.' {
                is_float = true;
            }
            self.advance();
            while self.peek().is_digit(10) {
                self.advance();
            }
        }
        if is_float {
            self.add_token(TokenKind::FloatLiteral);
        } else {
            self.add_token(TokenKind::Integer)
        }
        Ok(())
    }
    fn scan_token(&mut self) -> Result<(), Error> {
        let c = self.advance();
        match c {
            '(' => self.add_token(TokenKind::LParen),
            ')' => self.add_token(TokenKind::RParen),
            ':' => self.add_token(TokenKind::Colon),
            ',' => self.add_token(TokenKind::Comma),
            '=' => self.add_token(TokenKind::Equal),
            _ => {
                if c.is_digit(10) {
                    self.number()?;
                } else {
                    panic!("Unexpected character: {}", c);
                }
            }
        }
        Ok(())
    }

    fn scan_tokens(&mut self) -> Result<(), Error> {
        while !self.is_at_end() {
            self.start = self.current;
            self.scan_token()?;
        }
        self.add_token(TokenKind::Eof);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan_tokens(source: &str) -> Vec<Token> {
        let mut scanner = Scanner::new(source.to_string());
        let result = scanner.scan_tokens();
        assert!(result.is_ok());
        scanner.tokens
    }
    fn scan_token(source: &str) -> Token {
        let tokens = scan_tokens(source);
        tokens.first().unwrap().clone()
    }
    #[test]
    fn test_scanner() {
        let token = scan_token("42.5");
        assert_eq!(token.kind, TokenKind::FloatLiteral);
        assert_eq!(token.lexeme, "42.5");
        let token = scan_token("42");
        assert_eq!(token.kind, TokenKind::Integer);
        assert_eq!(token.lexeme, "42");

        // let tokens = scan_tokens("42.5 42");
        // assert_eq!(tokens.len(), 3);
        // assert_eq!(tokens[0].kind, TokenKind::FloatLiteral);
        // assert_eq!(tokens[0].lexeme, "42.5");
        // assert_eq!(tokens[1].kind, TokenKind::Integer);
        // assert_eq!(tokens[1].lexeme, "42");
    }
}
