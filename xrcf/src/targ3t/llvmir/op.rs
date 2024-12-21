use crate::dialect::func::Call;
use crate::dialect::func::Func;
use crate::ir::display_region_inside_func;
use crate::ir::Attribute;
use crate::ir::Block;
use crate::ir::BlockArgumentName;
use crate::ir::BlockName;
use crate::ir::Op;
use crate::ir::OpOperand;
use crate::ir::OpOperands;
use crate::ir::Operation;
use crate::ir::OperationName;
use crate::ir::Type;
use crate::ir::Value;
use crate::shared::Shared;
use crate::shared::SharedExt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

/// Display an operand LLVMIR style (e.g., `i32 8`, `i32 %0`, or `label %exit`).
fn display_operand(f: &mut Formatter<'_>, operand: &Shared<OpOperand>) -> std::fmt::Result {
    match &*operand.rd().value().rd() {
        Value::BlockArgument(block_arg) => {
            let name = block_arg.name();
            let name = name.rd();
            let name = match &*name {
                BlockArgumentName::Anonymous => panic!("Expected a named block argument"),
                BlockArgumentName::Name(name) => name.to_string(),
                BlockArgumentName::Unset => panic!("Block argument has no name"),
            };
            write!(f, "{name}")
        }
        Value::Constant(constant) => {
            let value = constant.value();
            let value = value.value();
            let typ = constant.typ();
            let typ = typ.rd();
            write!(f, "{typ} {value}")
        }
        Value::BlockLabel(label) => {
            panic!("BlockLabel for {label} should have been BlockPtr");
        }
        Value::BlockPtr(ptr) => {
            let label = match &*ptr.block().rd().label().rd() {
                BlockName::Name(name) => name.to_string(),
                BlockName::Unnamed => panic!("Expected a named block"),
                BlockName::Unset => panic!("Expected a named block"),
            };
            write!(f, "label %{label}")
        }
        Value::OpResult(op_result) => {
            let name = op_result.new_name();
            op_result.set_name(&name);
            write!(f, "{} {name}", op_result.typ().expect("no type").rd())
        }
        _ => panic!(
            "Unexpected operand value type for {}",
            operand.rd().value().rd()
        ),
    }
}

fn display_operands(f: &mut Formatter<'_>, operands: &OpOperands) -> std::fmt::Result {
    for (i, operand) in operands.vec().rd().iter().enumerate() {
        if 0 < i {
            write!(f, ", ")?;
        }
        display_operand(f, operand)?;
    }
    Ok(())
}

/// `add`
pub struct AddOp {
    operation: Shared<Operation>,
}

impl Op for AddOp {
    fn operation_name() -> OperationName {
        OperationName::new("target::llvmir::add".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        AddOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        let operation = self.operation().rd();
        write!(f, "{} = add ", operation.results())?;
        display_operands(f, &operation.operands())?;
        writeln!(f)
    }
}

impl Display for AddOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}

/// `alloca`
pub struct AllocaOp {
    operation: Shared<Operation>,
    element_type: Option<String>,
}

impl AllocaOp {
    pub fn array_size(&self) -> Arc<dyn Attribute> {
        match &*self.operation().rd().operand(0).unwrap().rd().value().rd() {
            Value::Constant(constant) => constant.value().clone(),
            _ => panic!("Unexpected"),
        }
    }
    pub fn set_element_type(&mut self, element_type: String) {
        self.element_type = Some(element_type);
    }
}

impl Op for AllocaOp {
    fn operation_name() -> OperationName {
        OperationName::new("alloca".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        AllocaOp {
            operation,
            element_type: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{} = ", self.operation().rd().results())?;
        write!(f, "{} ", AllocaOp::operation_name())?;
        write!(f, "{}", self.element_type.as_ref().unwrap())?;
        let array_size = self.array_size();
        let typ = array_size.typ();
        let typ = typ.rd();
        let value = array_size.value();
        write!(f, ", {typ} {value}, align 1")?;
        Ok(())
    }
}

impl Display for AllocaOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}

/// `call`
pub struct CallOp {
    operation: Shared<Operation>,
    identifier: Option<String>,
    varargs: Option<Shared<dyn Type>>,
}

impl Call for CallOp {
    fn identifier(&self) -> Option<String> {
        self.identifier.clone()
    }
    fn set_identifier(&mut self, identifier: String) {
        self.identifier = Some(identifier);
    }
    fn varargs(&self) -> Option<Shared<dyn Type>> {
        self.varargs.clone()
    }
    fn set_varargs(&mut self, varargs: Option<Shared<dyn Type>>) {
        self.varargs = varargs;
    }
}

impl Op for CallOp {
    fn operation_name() -> OperationName {
        OperationName::new("call".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        CallOp {
            operation,
            identifier: None,
            varargs: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        let operation = self.operation().rd();
        let results = operation.results();
        let n_results = results.vec().rd().len();
        if 0 < n_results {
            write!(f, "{} = ", results)?;
        }
        let return_type = if n_results == 1 {
            operation.result_type(0).unwrap().rd().to_string()
        } else {
            "void".to_string()
        };
        write!(f, "call {return_type} ")?;
        if let Some(varargs) = self.varargs() {
            write!(f, "({}) ", varargs.rd())?;
        }
        write!(f, "{}(", self.identifier().unwrap())?;
        display_operands(f, &operation.operands())?;
        write!(f, ")")?;
        Ok(())
    }
}

impl Display for CallOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}

/// `br`
///
/// Can be a conditional as well as an unconditional branch. Branch targets
/// are stored as operands.
pub struct BranchOp {
    operation: Shared<Operation>,
}

impl Op for BranchOp {
    fn operation_name() -> OperationName {
        OperationName::new("branch".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        BranchOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "br ")?;
        display_operands(f, &self.operation().rd().operands())?;
        Ok(())
    }
}

impl Display for BranchOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}

pub struct FuncOp {
    operation: Shared<Operation>,
    identifier: Option<String>,
}

impl FuncOp {
    fn has_implementation(&self) -> bool {
        self.operation().rd().region().is_some()
    }
}

impl Op for FuncOp {
    fn operation_name() -> OperationName {
        OperationName::new("target::llvmir::func".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        FuncOp {
            operation,
            identifier: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
    fn is_func(&self) -> bool {
        true
    }
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        let return_types = self.return_types();
        let return_type = if return_types.len() == 1 {
            return_types[0].rd().to_string()
        } else {
            "void".to_string()
        };
        let fn_keyword = if self.has_implementation() {
            "define"
        } else {
            "declare"
        };
        write!(
            f,
            "{fn_keyword} {return_type} {}(",
            self.identifier().unwrap()
        )?;
        let arguments = self.arguments().unwrap().into_iter();
        for (i, argument) in arguments.enumerate() {
            if 0 < i {
                write!(f, ", ")?;
            }
            match &*argument.rd() {
                Value::BlockArgument(arg) => {
                    let typ = arg.typ();
                    let typ = typ.rd();
                    match &*arg.name().rd() {
                        BlockArgumentName::Name(name) => write!(f, "{typ} {name}"),
                        BlockArgumentName::Anonymous => panic!("Expected a named block argument"),
                        BlockArgumentName::Unset => write!(f, "{}", typ),
                    }?;
                }
                Value::Variadic => write!(f, "...")?,
                _ => panic!("Unexpected"),
            }
        }
        write!(f, ")")?;

        let operation = self.operation().rd();
        display_region_inside_func(f, &*operation, indent)
    }
}

impl Func for FuncOp {
    fn identifier(&self) -> Option<String> {
        self.identifier.clone()
    }
    fn set_identifier(&mut self, identifier: String) {
        self.identifier = Some(identifier);
    }
}

impl Display for FuncOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}

pub struct ModuleOp {
    operation: Shared<Operation>,
    module_id: String,
    source_filename: String,
    module_flags: String,
    debug_info: String,
}

impl Op for ModuleOp {
    fn operation_name() -> OperationName {
        OperationName::new("llvm.mlir.module".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        ModuleOp {
            operation,
            module_id: "LLVMDialectModule".to_string(),
            source_filename: "LLVMDialectModule".to_string(),
            module_flags: "!{!0}".to_string(),
            debug_info: r#"!0 = !{i32 2, !"Debug Info Version", i32 3}"#.to_string(),
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        writeln!(f, "; ModuleID = '{}'", self.module_id)?;
        write!(f, r#"source_filename = "{}""#, self.source_filename)?;
        writeln!(f, "\n")?;
        if let Some(region) = self.operation().rd().region() {
            for block in region.rd().blocks().into_iter() {
                block.rd().display(f, indent)?;
            }
        }
        write!(f, "\n!llvm.module.flags = {}", self.module_flags)?;
        write!(f, "\n\n{}", self.debug_info)?;
        Ok(())
    }
}

impl Display for ModuleOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}

/// `phi`
pub struct PhiOp {
    operation: Shared<Operation>,
    argument_pairs: Option<Vec<(Shared<OpOperand>, Shared<Block>)>>,
}

impl Op for PhiOp {
    fn operation_name() -> OperationName {
        OperationName::new("phi".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        PhiOp {
            operation,
            argument_pairs: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{} = ", self.operation.rd().results())?;
        write!(f, "phi ")?;
        let pairs = self.argument_pairs().unwrap();
        assert!(pairs.len() == 2, "Expected two callers");
        write!(f, "{} ", pairs[0].0.rd().typ().unwrap().rd())?;
        let mut texts = vec![];
        for (value, block) in pairs {
            let value = value.rd();
            let value = value.value();
            let value = value.rd();
            let value = if let Value::Constant(constant) = &*value {
                // Drop type information.
                constant.value().value()
            } else {
                value.to_string()
            };
            let block = block.rd();
            let label = block.label();
            let label = label.rd();
            let mut label = match &*label {
                BlockName::Name(name) => name.to_string(),
                BlockName::Unnamed => panic!("Expected a named block"),
                BlockName::Unset => panic!("Expected a named block"),
            };
            if !label.starts_with('%') {
                label = format!("%{label}");
            }
            texts.push(format!("[ {value}, {label} ]"));
        }
        write!(f, "{}", texts.join(", "))?;
        Ok(())
    }
}

impl PhiOp {
    pub fn argument_pairs(&self) -> Option<Vec<(Shared<OpOperand>, Shared<Block>)>> {
        self.argument_pairs.clone()
    }
    pub fn set_argument_pairs(
        &mut self,
        argument_pairs: Option<Vec<(Shared<OpOperand>, Shared<Block>)>>,
    ) {
        self.argument_pairs = argument_pairs;
    }
}

/// `ret`
pub struct ReturnOp {
    operation: Shared<Operation>,
}

impl Op for ReturnOp {
    fn operation_name() -> OperationName {
        OperationName::new("ret".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        ReturnOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        let operands = self.operation().rd().operands();
        let name = Self::operation_name();
        if operands.vec().rd().is_empty() {
            write!(f, "{name} void")
        } else {
            write!(f, "{name} ")?;
            display_operands(f, &operands)
        }
    }
}

impl Display for ReturnOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}

/// `store`
pub struct StoreOp {
    operation: Shared<Operation>,
    len: Option<usize>,
}

impl StoreOp {
    pub fn value(&self) -> Shared<OpOperand> {
        self.operation.rd().operand(0).unwrap()
    }
    pub fn addr(&self) -> Shared<OpOperand> {
        self.operation.rd().operand(1).unwrap()
    }
    pub fn set_len(&mut self, len: usize) {
        self.len = Some(len);
    }
    pub fn len(&self) -> Option<usize> {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len() == Some(0)
    }
}

impl Op for StoreOp {
    fn operation_name() -> OperationName {
        OperationName::new("store".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        StoreOp {
            operation,
            len: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "store ")?;
        let len = self.len().expect("len not set during lowering");
        write!(f, "[{len} x i8] ")?;
        write!(f, "c{}, ", self.value().rd())?;
        write!(f, "ptr {}, ", self.addr().rd())?;
        write!(f, "align 1")?;
        Ok(())
    }
}

impl Display for StoreOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}
