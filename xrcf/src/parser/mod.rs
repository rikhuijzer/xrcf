//! Parsing logic for the compiler including a scanner (tokenizer).

mod parser;
mod scanner;
mod token;

pub use parser::default_dispatch;
pub use parser::DefaultParserDispatch;
pub use parser::Parse;
pub use parser::Parser;
pub use parser::ParserDispatch;
pub use token::Token;
pub use token::TokenKind;
