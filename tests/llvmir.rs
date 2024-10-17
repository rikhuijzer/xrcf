#![allow(dead_code)]
#![allow(unused)]
extern crate xrcf;
mod tester;

use crate::tester::Test;
use std::panic::Location;
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
    let expected = indoc! {"
    module {
      llvm.func @main() -> i64 {
        %0 = llvm.mlir.constant(42 : i64) : i64
        llvm.return %0 : i64
      }
    }
    "};
    let flags = "--convert-func-to-llvm";
    Test::init_subscriber(tracing::Level::DEBUG);
    let caller = Location::caller();
    Test::opt(flags, src, expected, caller);

    let src = expected;
    let expected = indoc! {r#"
    define i64 @main() {
      ret i64 42
    }
    "#};
    let flags = "--convert-mlir-to-llvmir";
    Test::parse(src, src, caller);
    let module = Test::opt(flags, src, expected, caller);
    let module = module.try_read().unwrap();
    let module = module
        .as_any()
        .downcast_ref::<targ3t::llvmir::ModuleOp>()
        .unwrap();
    let repr = format!("{}", module);
}
