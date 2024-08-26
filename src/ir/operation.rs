
// Takes attributes
pub trait Operation {
    fn parse(input: &str) -> Option<Self>
    where
        Self: Sized;

    fn name(&self) -> &'static str;
    fn print(&self) -> String {
        self.name().to_string()
    }
}