//! Cranelift IR (CLIR) operations.

use crate::ir::Op;
use crate::ir::Operation;
use crate::ir::OperationName;
use crate::shared::Shared;
use crate::shared::SharedExt;
use cranelift::codegen::ir::Function;
use cranelift::codegen::Context;
use cranelift::prelude::FunctionBuilderContext;
use cranelift_jit::JITModule;
use cranelift_module::FuncId;
use cranelift_module::Module;
use std::fmt::Display;
use std::fmt::Formatter;

pub struct FuncOp {
    operation: Shared<Operation>,
    func: Function,
}

impl FuncOp {
    pub fn func(&self) -> Function {
        self.func.clone()
    }
    pub fn set_func(&mut self, func: Function) {
        self.func = func;
    }
}

impl Op for FuncOp {
    fn operation_name() -> OperationName {
        OperationName::new("target::clif::func".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        let func = Function::new();
        FuncOp { operation, func }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{}", self.func)
    }
}

impl Display for FuncOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}

#[allow(dead_code)]
pub struct ModuleOp {
    operation: Shared<Operation>,
    /// Context that is re-used accross multiple `FunctionBuilder` instances.
    builder_context: FunctionBuilderContext,
    /// Main Cranelift context.
    ///
    /// "Cranelift separates this from `Module` to allow for parallel
    /// compilation, with a context per thread" (not yet implemented).
    ///
    /// My take: So essentially I should first figure out the global context of
    /// the module and then I can compile the functions separately and gather
    /// them. That's probably why the methods on the Cranelift `Module` do not
    /// define functions but instead take existing functions and declare and
    /// define them.
    ctx: Context,
    module: Option<JITModule>,
    funcs: Vec<FuncId>,
}

impl ModuleOp {
    pub fn module(&self) -> Option<&JITModule> {
        self.module.as_ref()
    }
    pub fn set_module(&mut self, module: JITModule) {
        self.module = Some(module);
    }
    pub fn machine_code(&self) -> Result<*const u8, String> {
        let module = self.module.as_ref().expect("no module");
        let func_id = self.funcs.get(0).expect("no func");
        let code = module.get_finalized_function(*func_id);
        Ok(code)
    }
}

impl Op for ModuleOp {
    fn operation_name() -> OperationName {
        OperationName::new("target::clif::module".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        let module = None;
        let builder_context = FunctionBuilderContext::new();
        let ctx = Context::new();
        let funcs = vec![];
        ModuleOp {
            operation,
            builder_context,
            ctx,
            module,
            funcs,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "foo {}", self.operation.rd())
    }
}

impl Display for ModuleOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}
