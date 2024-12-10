use crate::ir::Block;
use crate::ir::BlockLabel;
use crate::ir::Constant;
use crate::ir::Op;
use crate::ir::Operation;
use crate::ir::Type;
use crate::ir::Value;
use crate::parser::Parser;
use crate::parser::ParserDispatch;
use crate::parser::TokenKind;
use anyhow::Result;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::RwLock;

pub struct OpOperand {
    value: Arc<RwLock<Value>>,
}

impl OpOperand {
    pub fn new(value: Arc<RwLock<Value>>) -> Self {
        OpOperand { value }
    }
    pub fn name(&self) -> String {
        let value = self.value.try_read().unwrap();
        value.name().expect("no name")
    }
    pub fn value(&self) -> Arc<RwLock<Value>> {
        self.value.clone()
    }
    pub fn set_value(&mut self, value: Arc<RwLock<Value>>) {
        self.value = value;
    }
    /// If this `OpOperand` is the result of an operation, return the operation
    /// that defines it.
    pub fn defining_op(&self) -> Option<Arc<RwLock<dyn Op>>> {
        let value = self.value();
        let value = &*value.try_read().unwrap();
        match value {
            Value::BlockArgument(_) => None,
            Value::BlockLabel(_) => None,
            Value::Constant(_) => None,
            Value::FuncResult(_) => todo!(),
            Value::OpResult(op_res) => op_res.defining_op(),
            Value::Variadic => None,
        }
    }
    pub fn typ(&self) -> Result<Arc<RwLock<dyn Type>>> {
        let value = self.value.try_read().unwrap();
        value.typ()
    }
}

impl Display for OpOperand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = self.value.try_read().unwrap();
        match &*value {
            Value::Constant(constant) => write!(f, "{constant}"),
            _ => write!(f, "{}", self.name()),
        }
    }
}

pub trait GuardedOpOperand {
    fn defining_op(&self) -> Option<Arc<RwLock<dyn Op>>>;
    fn typ(&self) -> Result<Arc<RwLock<dyn Type>>>;
    fn value(&self) -> Arc<RwLock<Value>>;
    fn set_value(&mut self, value: Arc<RwLock<Value>>);
}

impl GuardedOpOperand for Arc<RwLock<OpOperand>> {
    fn defining_op(&self) -> Option<Arc<RwLock<dyn Op>>> {
        let op = self.try_read().unwrap();
        op.defining_op()
    }
    fn typ(&self) -> Result<Arc<RwLock<dyn Type>>> {
        self.try_read().unwrap().typ()
    }
    fn value(&self) -> Arc<RwLock<Value>> {
        self.try_read().unwrap().value()
    }
    fn set_value(&mut self, value: Arc<RwLock<Value>>) {
        self.try_write().unwrap().set_value(value);
    }
}

#[derive(Clone)]
pub struct OpOperands {
    operands: Arc<RwLock<Vec<Arc<RwLock<OpOperand>>>>>,
}

impl OpOperands {
    pub fn vec(&self) -> Arc<RwLock<Vec<Arc<RwLock<OpOperand>>>>> {
        self.operands.clone()
    }
    pub fn from_vec(operands: Vec<Arc<RwLock<OpOperand>>>) -> Self {
        OpOperands {
            operands: Arc::new(RwLock::new(operands)),
        }
    }
    pub fn set_operand(&mut self, index: usize, operand: Arc<RwLock<OpOperand>>) {
        let mut operands = self.operands.try_write().unwrap();
        if operands.len() == index {
            operands.push(operand);
        } else {
            operands[index] = operand;
        }
    }
    pub fn display_with_types(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let operands = self.operands.try_read().unwrap();
        if !operands.is_empty() {
            let mut texts = vec![];
            for operand in operands.iter() {
                let operand = operand.try_read().unwrap();
                let typ = operand.typ().unwrap();
                let typ = typ.try_read().unwrap();
                texts.push(format!("{operand} : {typ}"));
            }
            write!(f, "{}", texts.join(", "))?;
        }
        Ok(())
    }
}

impl Default for OpOperands {
    fn default() -> Self {
        OpOperands {
            operands: Arc::new(RwLock::new(vec![])),
        }
    }
}

impl Display for OpOperands {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let joined = self
            .operands
            .try_read()
            .unwrap()
            .iter()
            .map(|o| o.try_read().unwrap().to_string())
            .collect::<Vec<String>>()
            .join(", ");
        write!(f, "{}", joined)
    }
}

impl<T: ParserDispatch> Parser<T> {
    /// Parse an OpOperand like %0, x, or "hello".
    ///
    /// `var_token_kind` should be [TokenKind::PercentIdentifier] in MLIR
    /// (e.g., `%x` in `%c = arith.addi %x, %y`), but other languages may use
    /// different syntax (e.g., Python would use `x` in `c = x + y`).
    pub fn parse_op_operand(
        &mut self,
        parent: Arc<RwLock<Block>>,
        var_token_kind: TokenKind,
    ) -> Result<Arc<RwLock<OpOperand>>> {
        let next = self.peek();
        if next.kind == var_token_kind {
            let identifier = self.expect(var_token_kind)?;
            let name = identifier.lexeme.clone();
            let block = parent.try_read().expect("no parent");
            let assignment = block.assignment(&name);
            let assignment = match assignment {
                Some(assignment) => assignment,
                None => {
                    let msg = "Expected assignment before use.";
                    let msg = self.error(&identifier, msg);
                    return Err(anyhow::anyhow!(msg));
                }
            };
            let operand = OpOperand::new(assignment);
            Ok(Arc::new(RwLock::new(operand)))
        } else if next.kind == TokenKind::CaretIdentifier {
            let identifier = self.expect(TokenKind::CaretIdentifier)?;
            let label = BlockLabel::new(identifier.lexeme.clone());
            let label = Value::BlockLabel(label);
            let label = Arc::new(RwLock::new(label));
            let operand = OpOperand::new(label);
            Ok(Arc::new(RwLock::new(operand)))
        } else if next.kind == TokenKind::String {
            let text = self.parse_string()?;
            let text = Arc::new(text);
            let text = Constant::new(text);
            let text = Value::Constant(text);
            let text = Arc::new(RwLock::new(text));
            let operand = OpOperand::new(text);
            Ok(Arc::new(RwLock::new(operand)))
        } else {
            let msg = "Expected operand.";
            let msg = self.error(&next, msg);
            return Err(anyhow::anyhow!(msg));
        }
    }
    /// Parse a single operand into the given operation.
    pub fn parse_op_operand_into(
        &mut self,
        parent: Arc<RwLock<Block>>,
        var_token_kind: TokenKind,
        operation: &mut Operation,
    ) -> Result<Arc<RwLock<OpOperand>>> {
        let operand = self.parse_op_operand(parent, var_token_kind)?;
        operation.set_operand(0, operand.clone());
        Ok(operand)
    }
    fn is_op_operand(&mut self, var_token_kind: TokenKind) -> bool {
        self.peek().kind == var_token_kind
            || self.peek().kind == TokenKind::String
            || self.peek().kind == TokenKind::CaretIdentifier
    }
    /// Parse %0, %1, %0, "hello", or nothing.
    ///
    /// Nothing is allowed because `hello()` is a valid function definition in
    /// most languages. Verifying that the number of operands is correct is a
    /// task for the caller.
    pub fn parse_op_operands(
        &mut self,
        parent: Arc<RwLock<Block>>,
        var_token_kind: TokenKind,
    ) -> Result<OpOperands> {
        let mut arguments = vec![];
        if self.is_op_operand(var_token_kind) {
            loop {
                let operand = self.parse_op_operand(parent.clone(), var_token_kind)?;
                arguments.push(operand);
                if self.check(TokenKind::Comma) {
                    let _comma = self.advance();
                    continue;
                } else {
                    break;
                }
            }
        }
        let operands = OpOperands {
            operands: Arc::new(RwLock::new(arguments)),
        };
        Ok(operands)
    }
    /// Parse %0, %1, %0, "hello", or nothing into the given operation.
    pub fn parse_op_operands_into(
        &mut self,
        parent: Arc<RwLock<Block>>,
        var_token_kind: TokenKind,
        operation: &mut Operation,
    ) -> Result<OpOperands> {
        let operands = self.parse_op_operands(parent, var_token_kind)?;
        operation.set_operands(operands.clone());
        Ok(operands)
    }
}
