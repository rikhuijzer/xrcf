use crate::attribute::Attributes;

// Takes attributes
pub trait Op {
    fn new(name: &'static str, attrs: Attributes) -> Self
    where
        Self: Sized;
    fn parse(input: &str) -> Option<Self>
    where
        Self: Sized;

    fn name(&self) -> &'static str;
    fn print(&self) -> String {
        self.name().to_string()
    }
}