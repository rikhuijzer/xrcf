use crate::canonicalize::CanonicalizeResult;
use crate::ir::operation;
use crate::ir::operation::Operation;
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
use std::any::Any;
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
    fn from_operation(_operation: Arc<RwLock<Operation>>) -> Result<Self> {
        todo!()
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
        let attributes = operation::Attributes::new();
        let integer = parser.expect(TokenKind::Integer)?;
        let value = integer.lexeme;

        let _colon = parser.expect(TokenKind::Colon)?;

        let num_bits = parser.expect(TokenKind::IntType)?;
        let typ = IntegerType::from_str(&num_bits.lexeme);
        let value = APInt::from_str(&num_bits.lexeme, &value);
        let integer = IntegerAttr::new(typ, value);
        attributes
            .map()
            .write()
            .unwrap()
            .insert("value".to_string(), Arc::new(integer));
        operation.set_attributes(attributes);
        let operation = Arc::new(RwLock::new(operation));
        Ok(Arc::new(RwLock::new(ConstantOp { operation })))
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

        let lhs = operands[0].clone();
        let lhs = lhs.read().unwrap().defining_op();
        let lhs = match lhs {
            Some(lhs) => lhs,
            None => return CanonicalizeResult::Unchanged,
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
