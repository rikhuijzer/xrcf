use std::fmt::Display;
use std::fmt::Formatter;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenKind {
    // Markers
    Eof,
    Error,
    CodeComplete,

    // Identifiers
    /// foo
    BareIdentifier,
    /// @foo
    AtIdentifier,
    /// %foo
    PercentIdentifier,

    // Literals
    /// 1.0
    FloatLiteral,
    /// 42
    Integer,
    /// "foo"
    String,
    /// i4, si8, ui16
    IntType,

    // Punctuation
    /// ->
    Arrow,
    /// :
    Colon,
    /// '
    SingleQuote,
    /// ,
    Comma,
    /// =
    Equal,
    /// (
    LParen,
    /// )
    RParen,
    /// {
    LBrace,
    /// }
    RBrace,
    /// -
    Minus,
    /// !
    Exclamation,
    /// >
    Greater,
    /// <
    Less,

    // Keywords
    KwF16,
    KwF32,
    KwF64,
    KwF80,
    KwTrue,
    KwType,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Location {
    /// The line number of the token.
    line: usize,
    /// The column number of the token.
    column: usize,
    /// The character location in the raw source string.
    start: usize,
}

impl Display for Location {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "loc(:{}:{})", self.line, self.column)
    }
}

impl Location {
    pub fn new(line: usize, column: usize, start: usize) -> Self {
        Self {
            line,
            column,
            start,
        }
    }
    pub fn line(&self) -> usize {
        self.line
    }
    pub fn column(&self) -> usize {
        self.column
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Token {
    /// The kind of token, such as `@foo` (AtIdentifier) or `i64` (IntType).
    pub kind: TokenKind,
    /// The lexeme of the token, such as `@foo` (AtIdentifier) or `i64` (IntType).
    pub lexeme: String,
    pub location: Location,
}

impl Token {
    pub fn new(kind: TokenKind, lexeme: String, location: Location) -> Self {
        Self {
            kind,
            lexeme,
            location,
        }
    }
    pub fn line(&self) -> usize {
        self.location.line()
    }
    pub fn column(&self) -> usize {
        self.location.column()
    }
}

impl Display for Token {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} \"{}\" {}", self.kind, self.lexeme, self.location)
    }
}
