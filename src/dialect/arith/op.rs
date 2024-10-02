use crate::canonicalize::CanonicalizeResult;
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
        Ok(())
    }
    fn from_operation(operation: Arc<RwLock<Operation>>) -> Result<Self> {
        let op = ConstantOp { operation };
        op.verify()?;
        Ok(op)
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
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
        let operation = self.operation.write().unwrap();
        let attributes = operation.attributes();
        let attributes = attributes.map();
        let mut attributes = attributes.write().unwrap();
        attributes.insert("value".to_string(), value);
    }
}

impl<T: Parse> Parser<T> {
    /// Parse a type definition (e.g., `: i64`).
    fn typ(&mut self) -> Result<Type> {
        let _colon = self.expect(TokenKind::Colon)?;
        let typ = self.expect(TokenKind::IntType)?;
        let typ = Type::new(typ.lexeme.clone());
        Ok(typ)
    }
}

impl Parse for ConstantOp {
    fn op<T: Parse>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let mut operation = Operation::default();
        operation.set_results(parser.results()?);

        let operation_name = parser.expect(TokenKind::BareIdentifier)?;
        assert!(operation_name.lexeme == "arith.constant");
        operation.set_name(ConstantOp::operation_name());
        operation.set_parent(parent.clone());
        let integer = parser.expect(TokenKind::Integer)?;
        let value = integer.lexeme;

        let _colon = parser.expect(TokenKind::Colon)?;

        let num_bits = parser.expect(TokenKind::IntType)?;
        let typ = IntegerType::from_str(&num_bits.lexeme);
        let value = APInt::from_str(&num_bits.lexeme, &value);
        let integer = IntegerAttr::new(typ, value);
        let integer = Arc::new(integer);
        let mut attributes = operation.attributes();
        attributes.insert("value", integer);
        let operation = Arc::new(RwLock::new(operation));
        let op = ConstantOp::from_operation(operation);
        let op = match op {
            Ok(op) => op,
            Err(err) => {
                return Err(anyhow::anyhow!(err));
            }
        };
        Ok(Arc::new(RwLock::new(op)))
    }
}

pub struct AddiOp {
    operation: Arc<RwLock<Operation>>,
}

impl AddiOp {
    /// Canonicalize `addi(addi(x, c0), c1) -> addi(x, c0 + c1)`.
    fn addi_add_constant(&mut self) -> CanonicalizeResult {
        let operation = self.operation.read().unwrap();
        let operands = operation.operands();
        let operands = operands.read().unwrap();
        assert!(operands.len() == 2);

        println!("Looking up {}", self.operation().read().unwrap());

        let lhs = operands[0].clone();
        let lhs = lhs.read().unwrap();
        let lhs = match lhs.defining_op() {
            Some(lhs) => lhs,
            None => {
                println!("here 2");
                return CanonicalizeResult::Unchanged;
            }
        };
        let lhs = lhs.read().unwrap();
        let lhs = match lhs.as_any().downcast_ref::<ConstantOp>() {
            Some(lhs) => lhs,
            None => return CanonicalizeResult::Unchanged,
        };

        let rhs = operands[1].clone();
        let rhs = rhs.read().unwrap().defining_op();
        let rhs = match rhs {
            Some(rhs) => rhs,
            None => return CanonicalizeResult::Unchanged,
        };
        let rhs = rhs.read().unwrap();
        let rhs = match rhs.as_any().downcast_ref::<ConstantOp>() {
            Some(rhs) => rhs,
            None => return CanonicalizeResult::Unchanged,
        };

        let lhs_value = lhs.attribute("value").unwrap();
        let lhs_value = lhs_value.as_any().downcast_ref::<IntegerAttr>().unwrap();
        let rhs_value = rhs.attribute("value").unwrap();
        let rhs_value = rhs_value.as_any().downcast_ref::<IntegerAttr>().unwrap();
        let new_value = IntegerAttr::add(lhs_value, rhs_value);

        let mut new_operation = Operation::default();
        new_operation.set_name(ConstantOp::operation_name());
        new_operation.set_parent(rhs.operation().read().unwrap().parent());

        let new_const = Arc::new(RwLock::new(new_operation));
        let new_const = match ConstantOp::from_operation(new_const) {
            Ok(new_const) => new_const,
            Err(err) => {
                panic!("{}", err);
            }
        };
        let new_const = Arc::new(RwLock::new(new_const));
        self.insert_before(new_const);

        CanonicalizeResult::Changed
    }
}

impl Op for AddiOp {
    fn operation_name() -> OperationName {
        OperationName::new("arith.addi".to_string())
    }
    fn from_operation(_operation: Arc<RwLock<Operation>>) -> Result<Self> {
        todo!()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn canonicalize(&mut self) -> CanonicalizeResult {
        self.addi_add_constant()
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{}", self.operation().read().unwrap())
    }
}

impl<T: Parse> Parser<T> {
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
    let mut results = results.write().unwrap();
    for result in results.iter_mut() {
        let mut mut_result = result.write().unwrap();
        match &mut *mut_result {
            Value::BlockArgument(_) => {
                panic!("This case should not occur")
            }
            Value::OpResult(res) => res.set_defining_op(Some(op.clone())),
        }
    }
}

impl Parse for AddiOp {
    fn op<T: Parse>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let mut operation = Operation::default();
        let results = parser.results()?;
        operation.set_results(results.clone());

        let operation_name = parser.expect(TokenKind::BareIdentifier)?;
        assert!(operation_name.lexeme == "arith.addi");
        operation.set_name(AddiOp::operation_name());
        assert!(parent.is_some());
        operation.set_parent(parent.clone());
        operation.set_operands(parser.operands(parent.unwrap())?);
        let _colon = parser.expect(TokenKind::Colon)?;
        let result_type = parser.expect(TokenKind::IntType)?;
        let result_type = Type::new(result_type.lexeme.clone());
        operation.set_result_types(vec![result_type]);

        let operation = Arc::new(RwLock::new(operation));
        let op = AddiOp { operation };
        let op = Arc::new(RwLock::new(op));
        set_defining_op(results, op.clone());
        Ok(op)
    }
}

// In MLIR this works by taking an OpAsmParser and parsing
// the elements of the op.
// Parsing tries to cast the elements to the expected types.
// If all succeeds, the elements are parsed into the struct.
// todo!()
// }
// enum ArithOp {
//    Addi(Addi),
//}

impl Dialect for Arith {
    fn name(&self) -> &'static str {
        "arith"
    }

    fn description(&self) -> &'static str {
        "Arithmetic operations."
    }

    // Probably we don't want to have a global obs state but instead
    // have some differrent implementations for common functions.
    // fn ops(&self) ->
    // }
}
