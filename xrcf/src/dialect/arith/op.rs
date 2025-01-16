use crate::convert::ChangedOp;
use crate::convert::RewriteResult;
use crate::frontend::Parse;
use crate::frontend::Parser;
use crate::frontend::ParserDispatch;
use crate::frontend::TokenKind;
use crate::ir::AnyType;
use crate::ir::Attribute;
use crate::ir::Block;
use crate::ir::IntegerAttr;
use crate::ir::IntegerType;
use crate::ir::Op;
use crate::ir::OpResult;
use crate::ir::Operation;
use crate::ir::OperationName;
use crate::ir::Value;
use crate::ir::Values;
use crate::shared::Shared;
use crate::shared::SharedExt;
use anyhow::Result;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

/// The token kind used for variables in this dialect.
///
/// We have to set this variable in each dialect separately because different
/// dialects may use different token kinds for variables.
const TOKEN_KIND: TokenKind = TokenKind::PercentIdentifier;

pub struct AddiOp {
    operation: Shared<Operation>,
}

impl AddiOp {
    /// Canonicalize `addi(addi(x, c0), c1) -> addi(x, c0 + c1)`.
    fn addi_add_constant(&self) -> RewriteResult {
        let operands = self.operation.rd().operands();
        assert!(operands.clone().into_iter().len() == 2);

        let lhs = operands.clone().into_iter().next().unwrap();
        let lhs = match lhs.rd().defining_op() {
            Some(lhs) => lhs,
            None => {
                return RewriteResult::Unchanged;
            }
        };
        let lhs = lhs.rd();
        let lhs = match lhs.as_any().downcast_ref::<ConstantOp>() {
            Some(lhs) => lhs,
            None => return RewriteResult::Unchanged,
        };

        let rhs = operands.into_iter().nth(1).unwrap();
        let rhs = match rhs.rd().defining_op() {
            Some(rhs) => rhs,
            None => return RewriteResult::Unchanged,
        };
        let rhs = rhs.rd();
        let rhs = match rhs.as_any().downcast_ref::<ConstantOp>() {
            Some(rhs) => rhs,
            None => return RewriteResult::Unchanged,
        };

        let lhs_value = lhs.attribute("value").unwrap();
        let lhs_value = lhs_value.as_any().downcast_ref::<IntegerAttr>().unwrap();
        let rhs_value = rhs.attribute("value").unwrap();
        let rhs_value = rhs_value.as_any().downcast_ref::<IntegerAttr>().unwrap();
        let new_value = IntegerAttr::add(lhs_value, rhs_value);

        let mut new_operation = Operation::default();
        new_operation.set_name(ConstantOp::operation_name());
        new_operation.set_parent(rhs.operation().rd().parent());
        let attributes = rhs.operation().rd().attributes().deep_clone();
        attributes.insert("value", Arc::new(new_value));
        new_operation.set_attributes(attributes);

        let results = Values::default();
        let mut result = OpResult::default();
        result.set_name("%c3_i64");
        let result = Value::OpResult(result);
        let result = Shared::new(result.into());
        results.vec().wr().push(result.clone());
        new_operation.set_results(results);

        let new_const = ConstantOp::from_operation(new_operation);
        let new_const = Shared::new(new_const.into());
        if let Value::OpResult(result) = &mut *result.wr() {
            result.set_defining_op(Some(new_const.clone()));
        }

        self.replace(new_const.clone());

        new_const
            .rd()
            .operation()
            .rd()
            .results()
            .into_iter()
            .enumerate()
            .for_each(|(i, result)| {
                assert!(i == 0, "Expected exactly one result");
                result.wr().set_name("%c3_i64");
            });

        RewriteResult::Changed(ChangedOp::new(new_const))
    }
}

impl Op for AddiOp {
    fn operation_name() -> OperationName {
        OperationName::new("arith.addi".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        AddiOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn is_pure(&self) -> bool {
        true
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
    fn canonicalize(&self) -> RewriteResult {
        self.addi_add_constant()
    }
}

impl<T: ParserDispatch> Parser<T> {
    /// Parse a simple operation such as `arith.addi`.
    ///
    /// Parses an op that has one result and one or more operands.
    pub fn parse_simple_op<O: Op + 'static>(
        parser: &mut Parser<T>,
        parent: Option<Shared<Block>>,
    ) -> Result<Shared<O>> {
        let mut operation = Operation::default();
        assert!(parent.is_some());
        operation.set_parent(parent.clone());
        let result = parser.parse_op_result_into(TOKEN_KIND, &mut operation)?;
        parser.expect(TokenKind::Equal)?;
        parser.parse_operation_name_into::<O>(&mut operation)?;
        operation.set_operands(parser.parse_op_operands(parent.unwrap(), TOKEN_KIND)?);
        parser.expect(TokenKind::Colon)?;
        let result_type = parser.expect(TokenKind::IntType)?;
        let result_type = AnyType::new(&result_type.lexeme);
        let result_type = Shared::new(result_type.into());
        operation.set_result_type(0, result_type)?;

        let op = O::from_operation(operation);
        let op = Shared::new(op.into());
        result.set_defining_op(Some(op.clone()));
        Ok(op)
    }
}

impl Parse for AddiOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Shared<Block>>,
    ) -> Result<Shared<dyn Op>> {
        Ok(Parser::<T>::parse_simple_op::<AddiOp>(parser, parent)?)
    }
}

pub struct ConstantOp {
    operation: Shared<Operation>,
}

impl ConstantOp {
    pub fn value(&self) -> Arc<dyn Attribute> {
        let attributes = self.operation.rd().attributes();
        let value = attributes.get("value");
        let value = value.expect("no value for ConstantOp");
        value.clone()
    }
    pub fn set_value(&self, value: Arc<dyn Attribute>) {
        let attributes = self.operation.rd().attributes();
        attributes.insert("value", value);
    }
}

impl Op for ConstantOp {
    fn operation_name() -> OperationName {
        OperationName::new("arith.constant".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        ConstantOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn is_const(&self) -> bool {
        true
    }
    fn is_pure(&self) -> bool {
        true
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{} = ", self.operation.rd().results())?;
        write!(f, "{}", self.operation.rd().name())?;
        let value = self.value();
        write!(f, " {value}")?;
        Ok(())
    }
}

impl Parse for ConstantOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Shared<Block>>,
    ) -> Result<Shared<dyn Op>> {
        let mut operation = Operation::default();
        operation.set_parent(parent.clone());
        let results = parser.parse_op_results_into(TOKEN_KIND, &mut operation)?;
        parser.expect(TokenKind::Equal)?;
        parser.parse_operation_name_into::<ConstantOp>(&mut operation)?;

        let value = if parser.is_boolean() {
            let typ = IntegerType::new(1);
            let typ = Shared::new(typ.into());
            operation.set_result_type(0, typ)?;
            parser.parse_boolean()?
        } else {
            let integer = parser.parse_integer()?;
            let typ = integer
                .as_any()
                .downcast_ref::<IntegerAttr>()
                .unwrap()
                .typ();
            operation.set_result_type(0, typ)?;
            Arc::new(integer)
        };

        let operation = Shared::new(operation.into());
        let op = ConstantOp { operation };
        op.set_value(value);
        let op = Shared::new(op.into());
        results.set_defining_op(op.clone());
        Ok(op)
    }
}

impl Display for ConstantOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}

pub struct DivsiOp {
    operation: Shared<Operation>,
}

impl Op for DivsiOp {
    fn operation_name() -> OperationName {
        OperationName::new("arith.divsi".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        DivsiOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn is_pure(&self) -> bool {
        true
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
}

impl Parse for DivsiOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Shared<Block>>,
    ) -> Result<Shared<dyn Op>> {
        Ok(Parser::<T>::parse_simple_op::<DivsiOp>(parser, parent)?)
    }
}

pub struct SubiOp {
    operation: Shared<Operation>,
}

impl Op for SubiOp {
    fn operation_name() -> OperationName {
        OperationName::new("arith.subi".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        SubiOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn is_pure(&self) -> bool {
        true
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
}

impl Parse for SubiOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Shared<Block>>,
    ) -> Result<Shared<dyn Op>> {
        Ok(Parser::<T>::parse_simple_op::<SubiOp>(parser, parent)?)
    }
}
