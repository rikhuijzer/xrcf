use crate::ir::Attribute;
use crate::ir::StringType;
use crate::ir::Type;
use crate::parser::Parser;
use crate::parser::ParserDispatch;
use std::fmt::Formatter;
use std::fmt::Result;
use std::sync::Arc;
use std::sync::RwLock;
pub struct LinkageAttr {
    value: String,
}

impl Attribute for LinkageAttr {
    fn from_str(value: &str) -> Self {
        Self {
            value: value.to_string(),
        }
    }
    fn typ(&self) -> Arc<RwLock<dyn Type>> {
        Arc::new(RwLock::new(StringType::new()))
    }
    fn parse<T: ParserDispatch>(parser: &mut Parser<T>) -> Option<Self> {
        let next = parser.peek();
        if next.lexeme == "internal" {
            parser.advance();
            Some(Self::from_str("internal"))
        } else {
            Some(Self::from_str("external"))
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
