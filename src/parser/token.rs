#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenKind {
    // Markers
    Eof,
    Error,
    CodeComplete,

    // Identifiers
    BareIdentifier,    // foo
    AtIdentifier,      // @foo
    PercentIdentifier, // %foo

    // Literals
    FloatLiteral, // 1.0
    Integer,      // 42
    String,       // "foo"
    IntType,      // i4, si8, ui16

    // Punctuation
    Arrow,  // ->
    Colon,  // :
    Comma,  // ,
    Equal,  // =
    LParen, // (
    RParen, // )
    LBrace, // {
    RBrace, // }
    Minus,  // -

    // Keywords
    KwF16,
    KwF32,
    KwF64,
    KwF80,
    KwTrue,
    KwType,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Token {
    pub kind: TokenKind,
    pub lexeme: String,
    pub literal: Option<String>,
    pub position: usize,
}

impl Token {
    pub fn new(kind: TokenKind, lexeme: String, literal: Option<String>, position: usize) -> Self {
        Self {
            kind,
            lexeme,
            literal,
            position,
        }
    }
    pub fn print(&self) -> String {
        format!("{:?} {}", self.kind, self.lexeme)
    }
}
