use crate::convert::ChangedOp;
use crate::convert::RewriteResult;
use crate::ir::operation::Operation;
use crate::ir::Attribute;
use crate::ir::Block;
use crate::ir::IntegerAttr;
use crate::ir::Op;
use crate::ir::OpResult;
use crate::ir::OperationName;
use crate::ir::Type;
use crate::ir::Value;
use crate::ir::Values;
use crate::parser::Parse;
use crate::parser::Parser;
use crate::parser::ParserDispatch;
use crate::parser::TokenKind;
use crate::typ::APInt;
use crate::typ::IntegerType;
use crate::Dialect;
use anyhow::Result;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::RwLock;

struct Arith {}

pub struct ConstantOp {
    operation: Arc<RwLock<Operation>>,
}

impl Op for ConstantOp {
    fn operation_name() -> OperationName {
        OperationName::new("arith.constant".to_string())
    }
    fn verify(&self) -> Result<()> {
        let read_only = self.operation().read().unwrap();
        if read_only.name() != ConstantOp::operation_name() {
            return Err(anyhow::anyhow!(
                "Invalid operation name for ConstantOp:\n  {}",
                read_only
            ));
        }
        if read_only.parent().is_none() {
            return Err(anyhow::anyhow!(
                "Parent is none for ConstantOp:\n  {}",
                read_only
            ));
        }
        let attributes = read_only.attributes();
        let attributes = attributes.map();
        let attributes = attributes.read().unwrap();
        if attributes.get("value").is_none() {
            return Err(anyhow::anyhow!(
                "Value is none for ConstantOp:\n  {}",
                read_only
            ));
        }
        let results = read_only.results();
        let results = results.read().unwrap();
        if results.len() != 1 {
            return Err(anyhow::anyhow!(
                "Results length for ConstantOp is not 1:\n  {}",
                read_only
            ));
        }
        Ok(())
    }
    fn from_operation_without_verify(
        operation: Arc<RwLock<Operation>>,
        name: OperationName,
    ) -> Result<Self> {
        operation.try_write().unwrap().set_name(name);
        Ok(ConstantOp { operation })
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn is_const(&self) -> bool {
        true
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{}", self.operation().read().unwrap())
    }
}

impl ConstantOp {
    fn value(&self) -> Arc<dyn Attribute> {
        let operation = self.operation.read().unwrap();
        let attributes = operation.attributes();
        let attributes = attributes.map();
        let attributes = attributes.read().unwrap();
        let value = attributes.get("value").unwrap();
        value.clone()
    }
    fn set_value(&mut self, value: Arc<dyn Attribute>) {
        let operation = self.operation.read().unwrap();
        let attributes = operation.attributes();
        let attributes = attributes.map();
        let mut attributes = attributes.write().unwrap();
        attributes.insert("value".to_string(), value);
    }
}

impl<T: ParserDispatch> Parser<T> {
    /// Parse a type definition (e.g., `: i64`).
    fn typ(&mut self) -> Result<Type> {
        let _colon = self.expect(TokenKind::Colon)?;
        let typ = self.expect(TokenKind::IntType)?;
        let typ = Type::new(typ.lexeme.clone());
        Ok(typ)
    }
    /// Parse a integer constant (e.g., `42 : i64`).
    pub fn integer(parser: &mut Parser<T>) -> Result<Arc<dyn Attribute>> {
        let integer = parser.expect(TokenKind::Integer)?;
        let value = integer.lexeme;

        let _colon = parser.expect(TokenKind::Colon)?;

        let num_bits = parser.expect(TokenKind::IntType)?;
        let typ = IntegerType::from_str(&num_bits.lexeme);
        let value = APInt::from_str(&num_bits.lexeme, &value);
        let integer = IntegerAttr::new(typ, value);
        Ok(Arc::new(integer))
    }
}

impl Parse for ConstantOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let mut operation = Operation::default();
        let results = parser.results()?;
        operation.set_results(results.clone());

        let operation_name = parser.expect(TokenKind::BareIdentifier)?;
        assert!(operation_name.lexeme == "arith.constant");
        operation.set_name(ConstantOp::operation_name());
        operation.set_parent(parent.clone());

        let integer = Parser::<T>::integer(parser)?;

        let attributes = operation.attributes();
        attributes.insert("value", integer);
        let operation = Arc::new(RwLock::new(operation));
        let op = ConstantOp {
            operation: operation.clone(),
        };
        let op = Arc::new(RwLock::new(op));
        set_defining_op(results, op.clone());
        Ok(op)
    }
}

pub struct AddiOp {
    operation: Arc<RwLock<Operation>>,
}

impl AddiOp {
    /// Canonicalize `addi(addi(x, c0), c1) -> addi(x, c0 + c1)`.
    fn addi_add_constant(&self) -> RewriteResult {
        let operation = self.operation.read().unwrap();
        let operands = operation.operands();
        let operands = operands.read().unwrap();
        assert!(operands.len() == 2);

        let lhs = operands[0].clone();
        let lhs = lhs.read().unwrap();
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

        let rhs = operands[1].clone();
        let rhs = rhs.read().unwrap().defining_op();
        let rhs = match rhs {
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
        new_operation.set_parent(rhs.operation().read().unwrap().parent());
        let attributes = rhs.operation().read().unwrap().attributes();
        let attributes = attributes.deep_clone();
        attributes.insert("value", Arc::new(new_value));
        new_operation.set_attributes(attributes);

        let results: Values = Arc::new(RwLock::new(vec![]));
        let mut result = OpResult::default();
        result.set_name("%c3_i64");
        let result = Value::OpResult(result);
        let result = Arc::new(RwLock::new(result));
        results.write().unwrap().push(result.clone());
        new_operation.set_results(results);

        let new_const = Arc::new(RwLock::new(new_operation));
        let new_const = match ConstantOp::from_operation(new_const) {
            Ok(new_const) => new_const,
            Err(err) => {
                panic!("{}", err);
            }
        };
        let new_const = Arc::new(RwLock::new(new_const));
        let mut result = result.try_write().unwrap();
        if let Value::OpResult(result) = &mut *result {
            result.set_defining_op(Some(new_const.clone()));
        }

        self.replace(new_const.clone());

        {
            let new_const = new_const.try_read().unwrap();
            let new_const = new_const.operation().try_read().unwrap();
            let results = new_const.results();
            let results = results.try_read().unwrap();
            assert!(results.len() == 1);
            let mut result = results[0].try_write().unwrap();
            result.rename("%c3_i64");
        }

        RewriteResult::Changed(ChangedOp::new(new_const))
    }
}

impl Op for AddiOp {
    fn operation_name() -> OperationName {
        OperationName::new("arith.addi".to_string())
    }
    fn from_operation_without_verify(
        operation: Arc<RwLock<Operation>>,
        name: OperationName,
    ) -> Result<Self> {
        operation.try_write().unwrap().set_name(name);
        Ok(AddiOp { operation })
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn canonicalize(&self) -> RewriteResult {
        self.addi_add_constant()
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{}", self.operation().read().unwrap())
    }
}

impl<T: ParserDispatch> Parser<T> {
    pub fn results(&mut self) -> Result<Values> {
        let mut results = vec![];
        while self.check(TokenKind::PercentIdentifier) {
            let identifier = self.expect(TokenKind::PercentIdentifier)?;
            let name = identifier.lexeme.clone();
            let mut op_result = OpResult::default();
            op_result.set_name(&name);
            let result = Value::OpResult(op_result);
            results.push(Arc::new(RwLock::new(result)));
            if self.check(TokenKind::Equal) {
                let _equal = self.advance();
            }
        }
        let results = Arc::new(RwLock::new(results));
        Ok(results)
    }
}

pub fn set_defining_op(results: Arc<RwLock<Vec<Arc<RwLock<Value>>>>>, op: Arc<RwLock<dyn Op>>) {
    let results = results.read().unwrap();
    for result in results.iter() {
        let mut mut_result = result.write().unwrap();
        match &mut *mut_result {
            Value::BlockArgument(_) => {
                panic!("This case should not occur")
            }
            Value::OpResult(res) => res.set_defining_op(Some(op.clone())),
        }
    }
}

impl<T: ParserDispatch> Parser<T> {
    pub fn parse_add<O: Op + 'static>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
        expected_name: OperationName,
    ) -> Result<Arc<RwLock<O>>> {
        let mut operation = Operation::default();
        let results = parser.results()?;
        operation.set_results(results.clone());

        let operation_name = parser.expect(TokenKind::BareIdentifier)?;
        assert!(operation_name.lexeme == expected_name.name());
        operation.set_name(expected_name.clone());
        assert!(parent.is_some());
        operation.set_parent(parent.clone());
        operation.set_operands(parser.operands(parent.unwrap())?);
        let _colon = parser.expect(TokenKind::Colon)?;
        let result_type = parser.expect(TokenKind::IntType)?;
        let result_type = Type::new(result_type.lexeme.clone());
        let result_type = Arc::new(RwLock::new(result_type));
        let result_types = Arc::new(RwLock::new(vec![result_type]));
        operation.set_result_types(result_types);

        let operation = Arc::new(RwLock::new(operation));
        let op = O::from_operation_without_verify(operation, expected_name)?;
        let op = Arc::new(RwLock::new(op));
        set_defining_op(results, op.clone());
        Ok(op)
    }
}

impl Parse for AddiOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let expected_name = AddiOp::operation_name();
        let op = Parser::<T>::parse_add::<AddiOp>(parser, parent, expected_name)?;
        Ok(op)
    }
}

impl Dialect for Arith {
    fn name(&self) -> &'static str {
        "arith"
    }

    fn description(&self) -> &'static str {
        "Arithmetic operations."
    }

    // Probably we don't want to have a global obs state but instead
    // have some different implementations for common functions.
    // fn ops(&self) ->
    // }
}
