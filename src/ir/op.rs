use crate::ir::Operation;
use anyhow::Result;
use std::fmt::Display;
use std::fmt::Formatter;
use std::pin::Pin;
use crate::Parse;
use crate::Parser;
use std::sync::Arc;
use crate::parser::TokenKind;

/// This is the trait that is implemented by all operations.
/// FuncOp, for example, will be implemented by various dialects.
/// Note that the parser will parse the tokens into an `Operation`
/// and MLIR would cast the `Operation` into a specific `Op` variant
/// such as `FuncOp`.
pub trait Op: Display {
    fn name() -> &'static str
    where
        Self: Sized;
    fn from_operation(operation: Pin<Box<Operation>>) -> Result<Self>
    where
        Self: Sized;
    fn operation(&self) -> &Pin<Box<Operation>>;
    fn display(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.operation())
    }
}

pub struct FuncOp {
    identifier: String,
}

impl Op for FuncOp {
    fn name() -> &'static str {
        "func.func"
    }
    fn from_operation(operation: Pin<Box<Operation>>) -> Result<Self> {
        todo!()
        // Ok(FuncOp { operation })
    }
    fn operation(&self) -> &Pin<Box<Operation>> {
        todo!()
    }
    fn display(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "func.func {}", self.identifier)
    }
}

impl Display for FuncOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f)
    }
}

impl Parse for FuncOp {
    fn op<T: Parse>(parser: &mut Parser<T>) -> Result<Arc<dyn Op>> {
        let identifier = parser.advance();
        if identifier.kind != TokenKind::AtIdentifier {
            return Err(anyhow::anyhow!("Expected identifier, got {:?}", identifier.kind));
        }

        // println!("{:?}", name.print());
        let op = FuncOp {
            identifier: identifier.lexeme.clone(),
        };
        Ok(Arc::new(op))
    }
}

#[cfg(test)]
mod tests {
    use crate::parser::Parser;
    use crate::parser::parser::BuiltinParse;

    #[test]
    fn parse_func() {
        let src = "
          func.func @test_addi(%arg0 : i64, %arg1 : i64) -> i64 {
            %0 = arith.addi %arg0, %arg1 : i64
            return %0 : i64
          }
        ";
        let op = Parser::<BuiltinParse>::parse(src).unwrap();

        assert!(false);
    }
}