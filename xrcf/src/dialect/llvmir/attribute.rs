use crate::parser::Parser;
use crate::parser::ParserDispatch;
use crate::Attribute;
use std::fmt::Formatter;
use std::fmt::Result;

pub struct LinkageAttr {
    #[allow(dead_code)]
    name: String,
    value: String,
}

impl Attribute for LinkageAttr {
    fn new(name: &str, value: &str) -> Self {
        Self {
            name: name.to_string(),
            value: value.to_string(),
        }
    }
    fn parse<T: ParserDispatch>(parser: &mut Parser<T>, name: &str) -> Option<Self> {
        let next = parser.peek();
        if next.lexeme == "internal" {
            parser.advance();
            Some(Self::new(name, "internal"))
        } else {
            Some(Self::new(name, "external"))
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn value(&self) -> String {
        self.value.clone()
    }
    fn display(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}", self.value)
    }
}