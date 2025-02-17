use crate::frontend::Parser;
use crate::frontend::ParserDispatch;
use crate::frontend::TokenKind;
use crate::ir::Block;
use crate::ir::BlockLabel;
use crate::ir::Constant;
use crate::ir::Op;
use crate::ir::Operation;
use crate::ir::Type;
use crate::ir::Value;
use crate::shared::Shared;
use crate::shared::SharedExt;
use anyhow::Result;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

pub struct OpOperand {
    pub value: Shared<Value>,
}

impl OpOperand {
    pub fn from_block(block: Shared<Block>) -> Self {
        let value = Value::from_block(block);
        let value = Shared::new(value.into());
        OpOperand { value }
    }
    pub fn new(value: Shared<Value>) -> Self {
        OpOperand { value }
    }
    pub fn name(&self) -> String {
        self.value.rd().name().expect("no name")
    }
    /// If this `OpOperand` is the result of an operation, return the operation
    /// that defines it.
    ///
    /// Returns `None` if the operand is not a [Value::OpResult].
    pub fn defining_op(&self) -> Option<Shared<dyn Op>> {
        match &*self.value.rd() {
            Value::BlockArgument(_) => None,
            Value::BlockLabel(_) => None,
            Value::BlockPtr(_) => None,
            Value::Constant(_) => None,
            Value::FuncResult(_) => todo!(),
            Value::OpResult(op_res) => op_res.defining_op(),
            Value::Variadic => None,
        }
    }
    pub fn display_with_type(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self} : {}", self.typ().unwrap().rd())
    }
    pub fn typ(&self) -> Result<Shared<dyn Type>> {
        self.value.rd().typ()
    }
}

impl Display for OpOperand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &*self.value.rd() {
            Value::Constant(constant) => write!(f, "{constant}"),
            _ => write!(f, "{}", self.name()),
        }
    }
}

#[derive(Clone)]
pub struct OpOperands {
    operands: Shared<Vec<Shared<OpOperand>>>,
}

impl IntoIterator for OpOperands {
    type Item = Shared<OpOperand>;
    type IntoIter = std::vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.operands.rd().clone().into_iter()
    }
}

impl OpOperands {
    pub fn vec(&self) -> Shared<Vec<Shared<OpOperand>>> {
        self.operands.clone()
    }
    pub fn from_vec(operands: Vec<Shared<OpOperand>>) -> Self {
        OpOperands {
            operands: Shared::new(operands.into()),
        }
    }
    pub fn set_operand(&mut self, index: usize, operand: Shared<OpOperand>) {
        let mut operands = self.operands.wr();
        if operands.len() == index {
            operands.push(operand);
        } else {
            operands[index] = operand;
        }
    }
    pub fn display_with_types(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let operands = self.operands.rd();
        if !operands.is_empty() {
            for (index, operand) in operands.iter().enumerate() {
                if 0 < index {
                    write!(f, ", ")?;
                }
                let operand = operand.rd();
                operand.display_with_type(f)?;
            }
        }
        Ok(())
    }
}

impl Default for OpOperands {
    fn default() -> Self {
        OpOperands {
            operands: Shared::new(vec![].into()),
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
            .map(|o| o.rd().to_string())
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
        parent: Shared<Block>,
        var_token_kind: TokenKind,
    ) -> Result<Shared<OpOperand>> {
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
            Ok(Shared::new(operand.into()))
        } else if next.kind == TokenKind::CaretIdentifier {
            let identifier = self.expect(TokenKind::CaretIdentifier)?;
            let label = BlockLabel::new(identifier.lexeme.clone());
            let label = Value::BlockLabel(label);
            let label = Shared::new(label.into());
            let operand = OpOperand::new(label);
            Ok(Shared::new(operand.into()))
        } else if next.kind == TokenKind::String {
            let text = self.parse_string()?;
            let text = Arc::new(text);
            let text = Constant::new(text);
            let text = Value::Constant(text);
            let text = Shared::new(text.into());
            let operand = OpOperand::new(text);
            Ok(Shared::new(operand.into()))
        } else {
            let msg = "Expected operand.";
            let msg = self.error(next, msg);
            return Err(anyhow::anyhow!(msg));
        }
    }
    /// Parse a single operand into the given operation.
    pub fn parse_op_operand_into(
        &mut self,
        parent: Shared<Block>,
        var_token_kind: TokenKind,
        operation: &mut Operation,
    ) -> Result<Shared<OpOperand>> {
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
        parent: Shared<Block>,
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
            operands: Shared::new(arguments.into()),
        };
        Ok(operands)
    }
    /// Parse %0, %1, %0, "hello", or nothing into the given operation.
    pub fn parse_op_operands_into(
        &mut self,
        parent: Shared<Block>,
        var_token_kind: TokenKind,
        operation: &mut Operation,
    ) -> Result<OpOperands> {
        let operands = self.parse_op_operands(parent, var_token_kind)?;
        operation.set_operands(operands.clone());
        Ok(operands)
    }
}
