use crate::convert::ChangedOp;
use crate::convert::RewriteResult;
use crate::ir::AnyType;
use crate::ir::Attribute;
use crate::ir::Block;
use crate::ir::GuardedOpOperand;
use crate::ir::GuardedOperation;
use crate::ir::GuardedValue;
use crate::ir::IntegerAttr;
use crate::ir::IntegerType;
use crate::ir::Op;
use crate::ir::OpResult;
use crate::ir::Operation;
use crate::ir::OperationName;
use crate::ir::Value;
use crate::ir::Values;
use crate::parser::Parse;
use crate::parser::Parser;
use crate::parser::ParserDispatch;
use crate::parser::TokenKind;
use crate::shared::RwLockExt;
use anyhow::Result;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::RwLock;

/// The token kind used for variables in this dialect.
///
/// We have to set this variable in each dialect separately because different
/// dialects may use different token kinds for variables.
const TOKEN_KIND: TokenKind = TokenKind::PercentIdentifier;

pub struct ConstantOp {
    operation: Arc<RwLock<Operation>>,
}

impl ConstantOp {
    pub fn value(&self) -> Arc<dyn Attribute> {
        let attributes = self.operation.attributes();
        let value = attributes.get("value");
        let value = value.expect("no value for ConstantOp");
        value.clone()
    }
    pub fn set_value(&self, value: Arc<dyn Attribute>) {
        let attributes = self.operation.attributes();
        attributes.insert("value", value);
    }
}

impl Op for ConstantOp {
    fn operation_name() -> OperationName {
        OperationName::new("arith.constant".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
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
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{} = ", self.operation.results())?;
        write!(f, "{}", self.operation.name())?;
        let value = self.value();
        write!(f, " {value}")?;
        Ok(())
    }
}

impl Parse for ConstantOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let mut operation = Operation::default();
        operation.set_parent(parent.clone());
        let results = parser.parse_op_results_into(TOKEN_KIND, &mut operation)?;
        parser.expect(TokenKind::Equal)?;
        parser.parse_operation_name_into::<ConstantOp>(&mut operation)?;

        let value = if parser.is_boolean() {
            let typ = IntegerType::new(1);
            let typ = Arc::new(RwLock::new(typ));
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

        let operation = Arc::new(RwLock::new(operation));
        let op = ConstantOp { operation };
        op.set_value(value);
        let op = Arc::new(RwLock::new(op));
        results.set_defining_op(op.clone());
        Ok(op)
    }
}

impl Display for ConstantOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}

pub struct AddiOp {
    operation: Arc<RwLock<Operation>>,
}

impl AddiOp {
    /// Canonicalize `addi(addi(x, c0), c1) -> addi(x, c0 + c1)`.
    fn addi_add_constant(&self) -> RewriteResult {
        let operands = self.operation.operands();
        let operands = operands.vec();
        let operands = operands.try_read().unwrap();
        assert!(operands.len() == 2);

        let lhs = operands.get(0).unwrap();
        let lhs = match lhs.defining_op() {
            Some(lhs) => lhs,
            None => {
                return RewriteResult::Unchanged;
            }
        };
        let lhs = lhs.read().unwrap();
        let lhs = match lhs.as_any().downcast_ref::<ConstantOp>() {
            Some(lhs) => lhs,
            None => return RewriteResult::Unchanged,
        };

        let rhs = operands.get(1).unwrap();
        let rhs = match rhs.defining_op() {
            Some(rhs) => rhs,
            None => return RewriteResult::Unchanged,
        };
        let rhs = rhs.read().unwrap();
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
        new_operation.set_parent(rhs.operation().parent());
        let attributes = rhs.operation().attributes().deep_clone();
        attributes.insert("value", Arc::new(new_value));
        new_operation.set_attributes(attributes);

        let results = Values::default();
        let result = OpResult::default();
        result.set_name("%c3_i64");
        let result = Value::OpResult(result);
        let result = Arc::new(RwLock::new(result));
        results.vec().wr().push(result.clone());
        new_operation.set_results(results);

        let new_const = ConstantOp::from_operation(new_operation);
        let new_const = Arc::new(RwLock::new(new_const));
        let mut result = result.wr();
        if let Value::OpResult(result) = &mut *result {
            result.set_defining_op(Some(new_const.clone()));
        }

        self.replace(new_const.clone());

        {
            let new_const = new_const.try_read().unwrap();
            let new_const = new_const.operation().try_read().unwrap();
            let results = new_const.results();
            let results = results.vec();
            let results = results.try_read().unwrap();
            assert!(results.len() == 1);
            results[0].rename("%c3_i64");
        }

        RewriteResult::Changed(ChangedOp::new(new_const))
    }
}

impl Op for AddiOp {
    fn operation_name() -> OperationName {
        OperationName::new("arith.addi".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        AddiOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn is_pure(&self) -> bool {
        true
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn canonicalize(&self) -> RewriteResult {
        self.addi_add_constant()
    }
}

impl<T: ParserDispatch> Parser<T> {
    pub fn parse_add<O: Op + 'static>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<O>>> {
        let mut operation = Operation::default();
        assert!(parent.is_some());
        operation.set_parent(parent.clone());
        let results = parser.parse_op_results_into(TOKEN_KIND, &mut operation)?;
        parser.expect(TokenKind::Equal)?;
        parser.parse_operation_name_into::<O>(&mut operation)?;
        operation.set_operands(parser.parse_op_operands(parent.unwrap(), TOKEN_KIND)?);
        let _colon = parser.expect(TokenKind::Colon)?;
        let result_type = parser.expect(TokenKind::IntType)?;
        let result_type = AnyType::new(&result_type.lexeme);
        let result_type = Arc::new(RwLock::new(result_type));
        operation.set_result_type(0, result_type)?;

        let op = O::from_operation(operation);
        let op = Arc::new(RwLock::new(op));
        results.set_defining_op(op.clone());
        Ok(op)
    }
}

impl Parse for AddiOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let op = Parser::<T>::parse_add::<AddiOp>(parser, parent)?;
        Ok(op)
    }
}
