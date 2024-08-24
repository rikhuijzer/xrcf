
/// Attributes belong to operations and can be used to, for example, specify
/// a SSA value.
pub trait Attribute {
    fn new(name: &'static str, value: &'static str) -> Self;
    fn parse(input: &str) -> Self;

    fn name(&self) -> &'static str;
    fn value(&self) -> &'static str;
    fn print(&self) -> String {
        format!("{} = {}", self.name(), self.value())
    }
}

pub trait Attributes {
    fn new(attrs: Vec<impl Attribute>) -> Self;
    fn attrs(&self) -> Vec<impl Attribute>;
    fn print(&self) -> String {
        self.attrs().iter().map(|attr| attr.print()).collect::<Vec<_>>().join(", ")
    }
}

// Takes attributes
pub trait Op {
    fn new(name: &'static str, attrs: impl Attributes) -> Self
    where
        Self: Sized;

    fn name(&self) -> &'static str;
    fn verify(&self) -> bool {
        true
    }
    fn print(&self) -> String {
        self.name().to_string()
    }
}
