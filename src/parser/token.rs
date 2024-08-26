#[derive(Debug, PartialEq, Eq)]
pub enum TokenKind {
    // Markers
    Eof,
    Error,
    CodeComplete,

    // Identifiers
    BareIdentifier, // foo
    AtIdentifier, // @foo

    // Literals
    FloatLiteral, // 1.0
    Integer, // 42
    String, // "foo"
    IntType, // i4, si8, ui16

    // Punctuation
    Colon, // :
    Comma, // ,
    Equal, // =
    LParen, // (
    RParen, // )

    // Keywords
    KwF16,
    KwF32,
    KwF64,
    KwF80,
    KwTrue,
    KwType,
}

pub struct Token {
    pub kind: TokenKind,
    pub lexeme: String,
    pub literal: Option<String>,
}

impl Token {
    pub fn print(&self) -> String {
        format!("{:?} {}", self.kind, self.lexeme)
    }
}
