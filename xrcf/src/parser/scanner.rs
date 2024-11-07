use crate::parser::token::Location;
use crate::parser::token::Token;
use crate::parser::token::TokenKind;
use anyhow::Result;

pub struct Scanner {
    source: String,
    tokens: Vec<Token>,
    start: usize,
    current: usize,
    line: usize,
    column: usize,
}

impl Scanner {
    fn new(source: String) -> Self {
        Scanner {
            source,
            tokens: Vec::new(),
            start: 0,
            current: 0,
            line: 0,
            column: 0,
        }
    }
    fn is_at_end(&self) -> bool {
        self.current >= self.source.len()
    }
    fn advance(&mut self) -> char {
        let c = self.source.chars().nth(self.current).unwrap();
        self.current += 1;
        self.column += 1;
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
    fn peek_n(&self, n: usize) -> char {
        if self.current + n >= self.source.len() {
            return '\0';
        }
        self.source.chars().nth(self.current + n).unwrap()
    }
    fn peek_word(&mut self, already_matched: Option<char>) -> String {
        let mut word = String::new();
        for i in 0..10 {
            let c = self.peek_n(i);
            match c {
                '0'..='9' | 'a'..='z' | 'A'..='Z' | '_' => {
                    word.push(c);
                }
                _ => {
                    break;
                }
            };
        }
        let word = if let Some(c) = already_matched {
            format!("{}{}", c, word)
        } else {
            word
        };
        word
    }
    fn add_token(&mut self, kind: TokenKind) {
        let lexeme = if kind == TokenKind::Eof {
            "".to_string()
        } else {
            self.source[self.start..self.current].to_string()
        };
        let diff = lexeme.len();
        let column = self.column - diff;
        let location = Location::new(self.line, column, self.start);
        self.tokens.push(Token::new(kind, lexeme, location));
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
        c.is_alphabetic() || c == '_' || c == '@' || c == '%'
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
            s if s.starts_with('%') => TokenKind::PercentIdentifier,
            _ => TokenKind::BareIdentifier,
        };
        self.add_token(kind);
        Ok(())
    }
    fn arrow_or_minus(&mut self) -> Result<()> {
        if self.peek() == '>' {
            self.advance();
            self.add_token(TokenKind::Arrow);
        } else {
            self.add_token(TokenKind::Minus);
        }
        Ok(())
    }
    fn is_int_type(word: &str) -> bool {
        let types = vec!["i1", "i4", "i8", "i16", "i32", "i64", "i128"];
        types.contains(&word)
    }
    fn int_type(&mut self, c: char) -> Result<()> {
        let word = self.peek_word(Some(c));
        if Scanner::is_int_type(&word) {
            for _ in 0..(word.len() - 1) {
                self.advance();
            }
            self.add_token(TokenKind::IntType);
        }
        Ok(())
    }
    fn is_int_type_start(&mut self, c: char) -> bool {
        let word = self.peek_word(Some(c));
        Scanner::is_int_type(&word)
    }
    fn string(&mut self) -> Result<()> {
        while self.peek() != '"' && !self.is_at_end() {
            self.advance();
        }
        if self.is_at_end() {
            return Err(anyhow::anyhow!("Unterminated string"));
        } else {
            // self.peek() == '"'
            self.advance();
            self.add_token(TokenKind::String);
        }
        Ok(())
    }
    fn scan_token(&mut self) -> Result<()> {
        let c = self.advance();
        match c {
            '(' => self.add_token(TokenKind::LParen),
            ')' => self.add_token(TokenKind::RParen),
            '{' => self.add_token(TokenKind::LBrace),
            '}' => self.add_token(TokenKind::RBrace),
            ':' => self.add_token(TokenKind::Colon),
            '\'' => self.add_token(TokenKind::SingleQuote),
            ',' => self.add_token(TokenKind::Comma),
            '.' => self.add_token(TokenKind::Dot),
            '=' => self.add_token(TokenKind::Equal),
            '!' => self.add_token(TokenKind::Exclamation),
            '>' => self.add_token(TokenKind::Greater),
            '<' => self.add_token(TokenKind::Less),
            ' ' | '\r' | '\t' => (),
            '\n' => {
                self.line += 1;
                self.column = 0;
            }
            '-' => self.arrow_or_minus()?,
            '"' => self.string()?,
            s if self.is_int_type_start(s) => self.int_type(s)?,
            s if s.is_digit(10) => self.number()?,
            s if Scanner::is_identifier_start(s) => self.identifier()?,
            _ => {
                let column = if self.column == 0 { 0 } else { self.column - 1 };
                let location = Location::new(self.line, column, self.start);
                return Err(anyhow::anyhow!(Self::error(
                    &self.source,
                    &location,
                    &format!("Scanning failed starting at: {}", c)
                )));
            }
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
    pub fn error(src: &str, loc: &Location, msg: &str) -> String {
        let lines = src.split('\n').collect::<Vec<&str>>();
        let n = loc.line();
        let prev_line = if n > 0 {
            let prev_n = n - 1;
            let prev = lines[prev_n];
            format!("\n{prev_n}  | {prev}")
        } else {
            "".to_string()
        };
        let line = lines[n];
        let line_num_width = 4 + n.to_string().len();
        let err_indent = " ".repeat(loc.column() + line_num_width);
        format!("```{prev_line}\n{n}  | {line}\n{err_indent}^ {msg}\n```")
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
        assert_eq!(token.location.line(), 0);
        assert_eq!(token.location.column(), 0);
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
        assert_eq!(tokens[6].kind, TokenKind::IntType);
        assert_eq!(tokens[6].lexeme, "i32");
        assert_eq!(tokens[7].kind, TokenKind::RParen);
        assert_eq!(tokens[8].kind, TokenKind::Colon);
        assert_eq!(tokens[9].kind, TokenKind::IntType);
        assert_eq!(tokens[9].lexeme, "i32");
        assert_eq!(tokens[10].kind, TokenKind::Eof);

        let tokens = Scanner::scan("42: i32").unwrap();
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].kind, TokenKind::Integer);
        assert_eq!(tokens[0].lexeme, "42");
        assert_eq!(tokens[1].kind, TokenKind::Colon);
        assert_eq!(tokens[2].kind, TokenKind::IntType);
        assert_eq!(tokens[2].lexeme, "i32");
        assert_eq!(tokens[3].kind, TokenKind::Eof);

        let tokens = Scanner::scan("arith.addi %arg0, %arg1 : i64\n").unwrap();
        assert_eq!(tokens.len(), 7);
        assert_eq!(tokens[0].kind, TokenKind::BareIdentifier);
        assert_eq!(tokens[0].lexeme, "arith.addi");
        assert_eq!(tokens[1].kind, TokenKind::PercentIdentifier);
        assert_eq!(tokens[1].lexeme, "%arg0");
        assert_eq!(tokens[2].kind, TokenKind::Comma);
        assert_eq!(tokens[3].kind, TokenKind::PercentIdentifier);
        assert_eq!(tokens[3].lexeme, "%arg1");
        assert_eq!(tokens[4].kind, TokenKind::Colon);
        assert_eq!(tokens[5].kind, TokenKind::IntType);
        assert_eq!(tokens[5].lexeme, "i64");

        let src = "module {\n  %1 = arith.addi %0, %0 : i32\n}";
        let mut scanner = Scanner::new(src.to_string());
        scanner.scan_tokens().unwrap();
        let tokens = &scanner.tokens;
        assert_eq!(tokens[4].lexeme, "arith.addi");
        assert_eq!(tokens[4].location.line(), 1);
        assert_eq!(tokens[4].location.column(), 7);

        let text = scanner.source;
        assert_eq!(text, src);

        let text = Scanner::error(src, &tokens[4].location, "test");
        println!("text:\n{}", text);
        let lines = text.split('\n').collect::<Vec<&str>>();
        assert_eq!(lines[0], "```");
        assert_eq!(lines[1], "0  | module {");
        assert_eq!(lines[2], "1  |   %1 = arith.addi %0, %0 : i32");
        assert_eq!(lines[3], "            ^ test");
        assert_eq!(lines[4], "```");

        let tokens = Scanner::scan(r#"foo = "hello""#).unwrap();
        assert_eq!(tokens[0].kind, TokenKind::BareIdentifier);
        assert_eq!(tokens[0].lexeme, "foo");
        assert_eq!(tokens[1].kind, TokenKind::Equal);
        assert_eq!(tokens[2].kind, TokenKind::String);
        assert_eq!(tokens[2].lexeme, r#""hello""#);
        assert_eq!(tokens[3].kind, TokenKind::Eof);
        assert_eq!(tokens.len(), 4);

        let tokens = Scanner::scan(r#"!llvm.array<14 x i8>"#).unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Exclamation);
        assert_eq!(tokens[1].kind, TokenKind::BareIdentifier);
        assert_eq!(tokens[1].lexeme, "llvm.array");
    }
}
