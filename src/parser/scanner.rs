use crate::parser::token::Token;
use crate::parser::token::TokenKind;
use anyhow::Result;

pub struct Scanner {
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
        let lexeme = if kind == TokenKind::Eof {
            "".to_string()
        } else {
            self.source[self.start..self.current].to_string()
        };
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
    fn number(&mut self) -> Result<()> {
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
    // Whether the character is a valid identifier start character.
    fn is_identifier_start(c: char) -> bool {
        c.is_alphabetic() || c == '_' || c == '@'
    }
    // Whether the character is a valid identifier character.
    fn is_identifier(c: char) -> bool {
        c.is_alphabetic() || c == '_' || c == '.' || c.is_digit(10)
    }
    // Scan identifiers and keywords.
    fn identifier(&mut self) -> Result<()> {
        while Scanner::is_identifier(self.peek()) {
            self.advance();
        }
        let lexeme = self.source[self.start..self.current].to_string();
        let kind = match lexeme.as_str() {
            "f16" => TokenKind::KwF16,
            "true" => TokenKind::KwTrue,
            s if s.starts_with('@') => TokenKind::AtIdentifier,
            _ => TokenKind::BareIdentifier,
        };
        self.add_token(kind);
        Ok(())
    }
    fn scan_token(&mut self) -> Result<()> {
        let c = self.advance();
        match c {
            '(' => self.add_token(TokenKind::LParen),
            ')' => self.add_token(TokenKind::RParen),
            ':' => self.add_token(TokenKind::Colon),
            ',' => self.add_token(TokenKind::Comma),
            '=' => self.add_token(TokenKind::Equal),
            ' ' | '\r' | '\t' => (),
            '\n' => self.line += 1,
            s if s.is_digit(10) => self.number()?,
            s if Scanner::is_identifier_start(s) => self.identifier()?,
            _ => panic!("Unexpected character: {}", c),
        }
        Ok(())
    }

    fn scan_tokens(&mut self) -> Result<()> {
        while !self.is_at_end() {
            self.start = self.current;
            self.scan_token()?;
        }
        self.add_token(TokenKind::Eof);
        Ok(())
    }
    pub fn scan(src: &str) -> Result<Vec<Token>> {
        let mut scanner = Scanner::new(src.to_string());
        scanner.scan_tokens()?;
        Ok(scanner.tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan_token(source: &str) -> Result<Token> {
        let tokens = Scanner::scan(source)?;
        Ok(tokens.first().unwrap().clone())
    }
    #[test]
    fn test_scanner() {
        let token = scan_token("42.5").unwrap();
        assert_eq!(token.kind, TokenKind::FloatLiteral);
        assert_eq!(token.lexeme, "42.5");
        let token = scan_token("42").unwrap();
        assert_eq!(token.kind, TokenKind::Integer);
        assert_eq!(token.lexeme, "42");

        let tokens = Scanner::scan("42.5 42").unwrap();
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].kind, TokenKind::FloatLiteral);
        assert_eq!(tokens[0].lexeme, "42.5");
        assert_eq!(tokens[1].kind, TokenKind::Integer);
        assert_eq!(tokens[1].lexeme, "42");
        assert_eq!(tokens[2].kind, TokenKind::Eof);

        let src = "llvm.mlir.global internal @i32_global(42 : i32) : i32";
        let tokens = Scanner::scan(src).unwrap();
        assert_eq!(tokens.len(), 11);
        assert_eq!(tokens[0].kind, TokenKind::BareIdentifier);
        assert_eq!(tokens[0].lexeme, "llvm.mlir.global");
        assert_eq!(tokens[1].kind, TokenKind::BareIdentifier);
        assert_eq!(tokens[1].lexeme, "internal");
        assert_eq!(tokens[2].kind, TokenKind::AtIdentifier);
        assert_eq!(tokens[2].lexeme, "@i32_global");
        assert_eq!(tokens[3].kind, TokenKind::LParen);
        assert_eq!(tokens[4].kind, TokenKind::Integer);
        assert_eq!(tokens[4].lexeme, "42");
        assert_eq!(tokens[5].kind, TokenKind::Colon);
        assert_eq!(tokens[6].kind, TokenKind::BareIdentifier);
        assert_eq!(tokens[6].lexeme, "i32");
        assert_eq!(tokens[7].kind, TokenKind::RParen);
        assert_eq!(tokens[8].kind, TokenKind::Colon);
        assert_eq!(tokens[9].kind, TokenKind::BareIdentifier);
        assert_eq!(tokens[9].lexeme, "i32");
        assert_eq!(tokens[10].kind, TokenKind::Eof);

        let tokens = Scanner::scan("42: i32").unwrap();
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].kind, TokenKind::Integer);
        assert_eq!(tokens[0].lexeme, "42");
        assert_eq!(tokens[1].kind, TokenKind::Colon);
        assert_eq!(tokens[2].kind, TokenKind::BareIdentifier);
        assert_eq!(tokens[2].lexeme, "i32");
        assert_eq!(tokens[3].kind, TokenKind::Eof);
    }
}