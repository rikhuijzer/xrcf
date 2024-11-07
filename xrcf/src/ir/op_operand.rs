use crate::ir::Block;
use crate::ir::Op;
use crate::ir::Operation;
use crate::ir::Type;
use crate::ir::Value;
use crate::parser::Parser;
use crate::parser::ParserDispatch;
use crate::parser::TokenKind;
use anyhow::Result;
use std::fmt::Display;
use std::sync::Arc;
use std::sync::RwLock;

pub struct OpOperand {
    pub value: Arc<RwLock<Value>>,
}

impl OpOperand {
    pub fn new(value: Arc<RwLock<Value>>) -> Self {
        OpOperand { value }
    }
    pub fn name(&self) -> String {
        let value = self.value.try_read().unwrap();
        value.name().unwrap()
    }
    pub fn value(&self) -> Arc<RwLock<Value>> {
        self.value.clone()
    }
    /// If this `OpOperand` is the result of an operation, return the operation
    /// that defines it.
    pub fn defining_op(&self) -> Option<Arc<RwLock<dyn Op>>> {
        let value = self.value();
        let value = &*value.read().unwrap();
        match value {
            Value::BlockArgument(_) => None,
            Value::FuncResult(_) => todo!(),
            Value::OpResult(op_res) => op_res.defining_op(),
            Value::Variadic(_) => None,
        }
    }
    pub fn typ(&self) -> Arc<RwLock<dyn Type>> {
        let value = self.value.try_read().unwrap();
        value.typ()
    }
}

impl Display for OpOperand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

pub trait GuardedOpOperand {
    fn defining_op(&self) -> Option<Arc<RwLock<dyn Op>>>;
    fn typ(&self) -> Arc<RwLock<dyn Type>>;
    fn value(&self) -> Arc<RwLock<Value>>;
}

impl GuardedOpOperand for Arc<RwLock<OpOperand>> {
    fn defining_op(&self) -> Option<Arc<RwLock<dyn Op>>> {
        let op = self.try_read().unwrap();
        op.defining_op()
    }
    fn typ(&self) -> Arc<RwLock<dyn Type>> {
        self.try_read().unwrap().typ()
    }
    fn value(&self) -> Arc<RwLock<Value>> {
        self.try_read().unwrap().value()
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
            .map(|o| o.try_read().unwrap().name())
            .collect::<Vec<String>>()
            .join(", ");
        write!(f, "{}", joined)
    }
}

impl<T: ParserDispatch> Parser<T> {
    /// Parse %0.
    pub fn parse_op_operand(
        &mut self,
        parent: Arc<RwLock<Block>>,
    ) -> Result<Arc<RwLock<OpOperand>>> {
        let identifier = self.expect(TokenKind::PercentIdentifier)?;
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
    }
    /// Parse %0 into an operand of the given operation.
    pub fn parse_op_operand_into(
        &mut self,
        parent: Arc<RwLock<Block>>,
        operation: &mut Operation,
    ) -> Result<Arc<RwLock<OpOperand>>> {
        let operand = self.parse_op_operand(parent)?;
        operation.set_operand(operand.clone());
        Ok(operand)
    }
    /// Parse %0, %1.
    pub fn parse_op_operands(&mut self, parent: Arc<RwLock<Block>>) -> Result<OpOperands> {
        let mut arguments = vec![];
        while self.check(TokenKind::PercentIdentifier) {
            let operand = self.parse_op_operand(parent.clone())?;
            arguments.push(operand);
            if self.check(TokenKind::Comma) {
                let _comma = self.advance();
            }
        }
        let operands = OpOperands {
            operands: Arc::new(RwLock::new(arguments)),
        };
        Ok(operands)
    }
    pub fn parse_op_operands_into(
        &mut self,
        parent: Arc<RwLock<Block>>,
        operation: &mut Operation,
    ) -> Result<OpOperands> {
        let operands = self.parse_op_operands(parent)?;
        operation.set_operands(operands.clone());
        Ok(operands)
    }
}
