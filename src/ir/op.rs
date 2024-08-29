use crate::ir::BlockArgument;
use crate::ir::Operation;
use crate::ir::Type;
use crate::ir::Value;
use crate::parser::TokenKind;
use crate::Parse;
use crate::Parser;
use anyhow::Result;
use std::fmt::Display;
use std::fmt::Formatter;
use std::pin::Pin;
use std::sync::Arc;

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
    operands: Vec<Value>,
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
        write!(f, "func.func {}(", self.identifier)?;
        let joined = self
            .operands
            .iter()
            .map(|o| o.to_string())
            .collect::<Vec<String>>()
            .join(", ");
        write!(f, "{}", joined)?;
        write!(f, ")")
    }
}

impl Display for FuncOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f)
    }
}

impl<T: Parse> Parser<T> {
    pub fn identifier(&mut self, kind: TokenKind) -> Result<String> {
        let identifier = self.advance();
        if identifier.kind != kind {
            return Err(anyhow::anyhow!(
                "Expected {:?}, got {:?}",
                kind,
                identifier.kind
            ));
        }
        Ok(identifier.lexeme.clone())
    }
    pub fn operand(&mut self) -> Result<Value> {
        let identifier = self.advance();
        if identifier.kind != TokenKind::PercentIdentifier {
            return Err(anyhow::anyhow!(
                "Expected '%identifier', got {:?}",
                identifier.kind
            ));
        }
        let name = identifier.lexeme.clone();
        let typ = if self.check(TokenKind::Colon) {
            self.advance();
            let typ = self.advance();
            Type::new(typ.lexeme.clone())
        } else {
            Type::new("any".to_string())
        };
        let arg = BlockArgument::new(name, typ);
        let operand: Value = Value::BlockArgument(arg);
        if self.check(TokenKind::Comma) {
            self.advance();
        }
        Ok(operand)
    }
    pub fn operands(&mut self) -> Result<Vec<Value>> {
        let mut operands = vec![];
        while self.check(TokenKind::PercentIdentifier) {
            operands.push(self.operand()?);
        }
        Ok(operands)
    }
}

impl Parse for FuncOp {
    fn op<T: Parse>(parser: &mut Parser<T>) -> Result<Arc<dyn Op>> {
        // Similar to `FuncOp::parse` in MLIR's `FuncOps.cpp`.
        let identifier = parser.identifier(TokenKind::AtIdentifier).unwrap();
        if !parser.check(TokenKind::LParen) {
            return Err(anyhow::anyhow!(
                "Expected '(', got {:?}",
                parser.peek().kind
            ));
        }
        parser.advance();
        let operands = parser.operands()?;

        // println!("{:?}", name.print());
        let op = FuncOp {
            identifier,
            operands,
        };
        Ok(Arc::new(op))
    }
}

#[cfg(test)]
mod tests {
    use crate::parser::parser::BuiltinParse;
    use crate::parser::Parser;

    #[test]
    fn parse_func() {
        let src = "
          func.func @test_addi(%arg0 : i64, %arg1 : i64) -> i64 {
            %0 = arith.addi %arg0, %arg1 : i64
            return %0 : i64
          }
        ";
        let module = Parser::<BuiltinParse>::parse(src).unwrap();
        let op = module.first_op().unwrap();
        let repr = format!("{}", op);
        assert_eq!(
            repr,
            "func.func @test_addi(%arg0 : i64, %arg1 : i64) -> i64"
        );
    }
}
