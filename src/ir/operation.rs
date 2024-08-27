
// Takes attributes
//
// Note that MLIR distinguishes between Operation and Op.
// Operation generically models all operations.
// Op is an interface for more specific operations.
// For example, `ConstantOp` does not take inputs and gives one output.
// `ConstantOp` does also not specify fields since they are accessed
// via a pointer to the `Operation`.
// In MLIR, a specific Op can be casted from an Operation.
pub trait Operation {
    fn parse(input: &str) -> Option<Self>
    where
        Self: Sized;

    fn name(&self) -> &'static str;
    fn print(&self) -> String {
        todo!() 
    }
}