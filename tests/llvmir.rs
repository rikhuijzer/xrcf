#![allow(dead_code)]
#![allow(unused)]
extern crate xrcf;

use std::sync::Arc;
use std::sync::RwLock;
use xrcf::convert::RewriteResult;
use xrcf::dialect::llvmir;
use xrcf::ir;
use xrcf::opt;
use xrcf::parser::BuiltinParse;
use xrcf::targ3t;
use xrcf::OptOptions;
use xrcf::Parser;

#[test]
fn test_translate() {
    let mlir = "llvm.mlir.global internal @i32_global(42: i32) : i32";
    let llvmir = "i32_global = internal global i32 42";

    // assert_eq!(translate(mlir), llvmir);
}

#[test]
fn test_constant_func() {
    use indoc::indoc;

    let src = indoc! {"
      func.func @main() -> i64 {
        %0 = arith.constant 42 : i64
        return %0 : i64
      }
    "};

    let module = Parser::<BuiltinParse>::parse(src).unwrap();
    let module = Arc::new(RwLock::new(module));
    println!("\n-- Before convert-func-to-llvm:\n{src}");
    // mlir-opt --convert-func-to-llvm tmp.mlir
    let mut options = OptOptions::default();
    options.set_convert_func_to_llvm(true);

    opt(module.clone(), options).unwrap();
    let module_clone = module.clone();
    let module_read = module_clone.try_read().unwrap();
    println!("\n-- After convert-func-to-llvm:\n{module_read}\n");
    let repr = format!("{}", module_read);
    let lines = repr.lines().collect::<Vec<&str>>();
    assert_eq!(lines[0], "module {");
    assert_eq!(lines[1], "  llvm.func @main() -> i64 {");
    assert_eq!(lines[2], "    %0 = llvm.mlir.constant(42 : i64) : i64");
    assert_eq!(lines[3], "    llvm.return %0 : i64");
    assert_eq!(lines[4], "  }");
    assert_eq!(lines[5], "}");

    let mut options = OptOptions::default();
    options.set_mlir_to_llvmir(true);
    let module = opt(module, options).unwrap();
    let module = match module {
        RewriteResult::Changed(changed) => changed.0,
        RewriteResult::Unchanged => panic!("expected change"),
    };
    let module = module.try_read().unwrap();
    let module = module
        .as_any()
        .downcast_ref::<targ3t::llvmir::ModuleOp>()
        .unwrap();
    let repr = format!("{}", module);

    println!("\n-- After convert-mlir-to-llvmir:\n{module}\n");
    let lines = repr.lines().collect::<Vec<&str>>();
    assert_eq!(lines[0], "; ModuleID = 'LLVMDialectModule'");
    assert_eq!(lines[1], r#"source_filename = "LLVMDialectModule""#);
    assert_eq!(lines[2], "");
    assert_eq!(lines[3], "define i64 @main() {");
    assert_eq!(lines[4], "  ret i64 42");
    assert_eq!(lines[5], "}");
    assert_eq!(lines[6], "");
    assert_eq!(lines[7], "!llvm.module.flags = !{!0}");
    assert_eq!(lines[8], "");
    assert_eq!(lines[9], r#"!0 = !{i32 2, !"Debug Info Version", i32 3}"#);
}
