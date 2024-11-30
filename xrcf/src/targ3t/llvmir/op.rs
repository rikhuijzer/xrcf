use crate::dialect::func::Call;
use crate::dialect::func::Func;
use crate::ir::display_region_inside_func;
use crate::ir::Attribute;
use crate::ir::Block;
use crate::ir::BlockDest;
use crate::ir::GuardedBlock;
use crate::ir::GuardedOpOperand;
use crate::ir::GuardedOperation;
use crate::ir::GuardedRegion;
use crate::ir::Op;
use crate::ir::OpOperand;
use crate::ir::OpOperands;
use crate::ir::Operation;
use crate::ir::OperationName;
use crate::ir::Type;
use crate::ir::Value;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::RwLock;

/// Display an operand LLVMIR style (e.g., `i32 8`, `i32 %0`, or `label %exit`).
fn display_operand(f: &mut Formatter<'_>, operand: &Arc<RwLock<OpOperand>>) -> std::fmt::Result {
    let value = operand.value();
    let value = value.try_read().unwrap();
    match &*value {
        Value::BlockArgument(block_arg) => {
            let name = block_arg.name().expect("Block argument has no name");
            write!(f, "{name}")
        }
        Value::Constant(constant) => {
            let value = constant.value();
            let value = value.value();
            let typ = constant.typ();
            let typ = typ.try_read().unwrap();
            write!(f, "{typ} {value}")
        }
        Value::BlockLabel(label) => {
            let label = label.name();
            let label = label.split_once('^').unwrap().1;
            write!(f, "label %{label}")
        }
        Value::OpResult(op_result) => {
            let name = op_result.name().expect("Op result has no name");
            let typ = op_result.typ().expect("No type");
            let typ = typ.try_read().unwrap();
            write!(f, "{typ} {name}")
        }
        _ => panic!("Unexpected operand value type for {value}"),
    }
}

fn display_operands(f: &mut Formatter<'_>, operands: &OpOperands) -> std::fmt::Result {
    let operands = operands.vec();
    let operands = operands.try_read().unwrap();
    for (i, operand) in operands.iter().enumerate() {
        if 0 < i {
            write!(f, ", ")?;
        }
        display_operand(f, operand)?;
    }
    Ok(())
}

pub struct AddOp {
    operation: Arc<RwLock<Operation>>,
}

impl Op for AddOp {
    fn operation_name() -> OperationName {
        OperationName::new("target::llvmir::add".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        AddOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        let operation = self.operation().try_read().unwrap();
        let results = operation.results();
        let results = results.vec();
        let results = results.try_read().unwrap();
        let result = results[0].try_read().unwrap();
        write!(f, "{result} = add ")?;
        display_operands(f, &operation.operands())?;
        write!(f, "\n")
    }
}

impl Display for AddOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}

pub struct AllocaOp {
    operation: Arc<RwLock<Operation>>,
    element_type: Option<String>,
}

impl AllocaOp {
    pub fn array_size(&self) -> Arc<dyn Attribute> {
        let operand = self.operation().operand(0).unwrap();
        let operand = operand.try_read().unwrap();
        let value = operand.value();
        let value = value.try_read().unwrap();
        match &*value {
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
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        AllocaOp {
            operation,
            element_type: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{} = ", self.operation().results())?;
        write!(f, "{} ", AllocaOp::operation_name())?;
        write!(f, "{}", self.element_type.as_ref().unwrap())?;
        let array_size = self.array_size();
        let typ = array_size.typ();
        let typ = typ.try_read().unwrap();
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

pub struct CallOp {
    operation: Arc<RwLock<Operation>>,
    identifier: Option<String>,
    varargs: Option<Arc<RwLock<dyn Type>>>,
}

impl Call for CallOp {
    fn identifier(&self) -> Option<String> {
        self.identifier.clone()
    }
    fn set_identifier(&mut self, identifier: String) {
        self.identifier = Some(identifier);
    }
    fn varargs(&self) -> Option<Arc<RwLock<dyn Type>>> {
        self.varargs.clone()
    }
    fn set_varargs(&mut self, varargs: Option<Arc<RwLock<dyn Type>>>) {
        self.varargs = varargs;
    }
}

impl Op for CallOp {
    fn operation_name() -> OperationName {
        OperationName::new("call".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        CallOp {
            operation,
            identifier: None,
            varargs: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        let operation = self.operation().try_read().unwrap();
        let results = operation.results();
        let n_results = results.vec().try_read().unwrap().len();
        if 0 < n_results {
            write!(f, "{} = ", results)?;
        }
        let return_type = if n_results == 1 {
            let return_type = operation.result_type(0);
            let return_type = return_type.unwrap();
            let return_type = return_type.try_read().unwrap().to_string();
            return_type
        } else {
            "void".to_string()
        };
        write!(f, "call {return_type} ")?;
        let varargs = self.varargs();
        if let Some(varargs) = varargs {
            write!(f, "({}) ", varargs.try_read().unwrap())?;
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

pub struct BranchOp {
    operation: Arc<RwLock<Operation>>,
    dest: Option<Arc<RwLock<BlockDest>>>,
}

impl BranchOp {
    pub fn set_dest(&mut self, dest: Arc<RwLock<BlockDest>>) {
        self.dest = Some(dest);
    }
}

impl Op for BranchOp {
    fn operation_name() -> OperationName {
        OperationName::new("branch".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        BranchOp {
            operation,
            dest: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "br ")?;
        if let Some(dest) = &self.dest {
            write!(f, "label {}", dest.try_read().unwrap())?;
        } else {
            // Conditional branch (e.g., `br i1 %cond, label %then, label %else`).
            display_operands(f, &self.operation().operands())?;
        }
        Ok(())
    }
}

impl Display for BranchOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}

pub struct FuncOp {
    operation: Arc<RwLock<Operation>>,
    identifier: Option<String>,
}

impl FuncOp {
    fn has_implementation(&self) -> bool {
        self.operation().region().is_some()
    }
}

impl Op for FuncOp {
    fn operation_name() -> OperationName {
        OperationName::new("target::llvmir::func".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        FuncOp {
            operation,
            identifier: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn is_func(&self) -> bool {
        true
    }
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        let return_types = self.return_types();
        let return_type = if return_types.len() == 1 {
            let return_type = return_types[0].clone();
            let return_type = return_type.try_read().unwrap();
            return_type.to_string()
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
        let arguments = self.arguments().unwrap();
        let arguments = arguments.vec();
        let arguments = arguments.try_read().unwrap();
        for (i, argument) in arguments.iter().enumerate() {
            if 0 < i {
                write!(f, ", ")?;
            }
            let argument = argument.try_read().unwrap();
            match &*argument {
                Value::BlockArgument(arg) => {
                    let typ = arg.typ();
                    let typ = typ.try_read().unwrap();
                    match arg.name() {
                        Some(name) => write!(f, "{} {}", typ, name),
                        None => write!(f, "{}", typ),
                    }?;
                }
                Value::Variadic => write!(f, "...")?,
                _ => panic!("Unexpected"),
            }
        }
        write!(f, ")")?;

        let operation = self.operation().try_read().unwrap();
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
    operation: Arc<RwLock<Operation>>,
    module_id: String,
    source_filename: String,
    module_flags: String,
    debug_info: String,
}

impl Op for ModuleOp {
    fn operation_name() -> OperationName {
        OperationName::new("llvm.mlir.module".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
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
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        write!(f, "; ModuleID = '{}'\n", self.module_id)?;
        write!(f, r#"source_filename = "{}""#, self.source_filename)?;
        write!(f, "\n\n")?;
        let region = self.operation().region();
        if let Some(region) = region {
            let blocks = region.blocks();
            let blocks = blocks.try_read().unwrap();
            for block in blocks.iter() {
                block.display(f, indent)?;
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

pub struct PhiOp {
    operation: Arc<RwLock<Operation>>,
    argument_pairs: Option<Vec<(Arc<RwLock<OpOperand>>, Arc<RwLock<Block>>)>>,
}

impl Op for PhiOp {
    fn operation_name() -> OperationName {
        OperationName::new("phi".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        PhiOp {
            operation,
            argument_pairs: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "phi ")?;
        let pairs = self.argument_pairs().unwrap();
        assert!(pairs.len() == 2, "Expected two callers");
        let typ = pairs[0].0.try_read().unwrap().typ().unwrap();
        let typ = typ.try_read().unwrap();
        write!(f, "{typ} ")?;
        let mut texts = vec![];
        for (value, block) in pairs {
            let value = value.try_read().unwrap();
            let value = value.value();
            let value = value.try_read().unwrap();
            let value = if let Value::Constant(constant) = &*value {
                // Drop type information.
                constant.value().value()
            } else {
                value.to_string()
            };
            let block = block.try_read().unwrap();
            let mut label = block.label().expect("expected label");
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
    pub fn argument_pairs(&self) -> Option<Vec<(Arc<RwLock<OpOperand>>, Arc<RwLock<Block>>)>> {
        self.argument_pairs.clone()
    }
    pub fn set_argument_pairs(
        &mut self,
        argument_pairs: Option<Vec<(Arc<RwLock<OpOperand>>, Arc<RwLock<Block>>)>>,
    ) {
        self.argument_pairs = argument_pairs;
    }
}

pub struct ReturnOp {
    operation: Arc<RwLock<Operation>>,
}

impl Op for ReturnOp {
    fn operation_name() -> OperationName {
        OperationName::new("ret".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        ReturnOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        let operands = self.operation().operands();
        let name = Self::operation_name();
        if operands.vec().try_read().unwrap().is_empty() {
            write!(f, "{name} void")
        } else {
            write!(f, "{name} ")?;
            display_operands(f, &self.operation.operands())
        }
    }
}

impl Display for ReturnOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}

pub struct StoreOp {
    operation: Arc<RwLock<Operation>>,
    len: Option<usize>,
}

impl StoreOp {
    pub fn addr(&self) -> Arc<RwLock<OpOperand>> {
        let operation = self.operation.try_read().unwrap();
        let operand = operation.operand(1).unwrap();
        operand
    }
    pub fn set_len(&mut self, len: usize) {
        self.len = Some(len);
    }
    pub fn len(&self) -> Option<usize> {
        self.len
    }
}

impl Op for StoreOp {
    fn operation_name() -> OperationName {
        OperationName::new("store".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        StoreOp {
            operation,
            len: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "store ")?;
        let const_value = self.operation().operand(0).unwrap();
        let const_value = const_value.try_read().unwrap();
        let len = self.len().expect("len not set during lowering");
        write!(f, "[{len} x i8] ")?;
        write!(f, "c{const_value}, ")?;
        write!(f, "ptr {}, ", self.addr().try_read().unwrap())?;
        write!(f, "align 1")?;
        Ok(())
    }
}

impl Display for StoreOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}
