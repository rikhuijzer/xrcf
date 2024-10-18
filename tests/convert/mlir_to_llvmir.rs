extern crate xrcf;

use crate::tester::Test;
use indoc::indoc;
use std::panic::Location;
use xrcf::targ3t;

const FLAGS: &str = "--convert-mlir-to-llvmir";

#[test]
fn test_constant() {
    let src = indoc! {"
    module {
      llvm.func @main() -> i64 {
        %0 = llvm.mlir.constant(42 : i64) : i64
        llvm.return %0 : i64
      }
    }
    "};

    let expected = indoc! {r#"
    ; ModuleID = 'LLVMDialectModule'
    source_filename = "LLVMDialectModule"

    define i64 @main() {
      ret i64 42
    }

    !llvm.module.flags = !{!0}

    !0 = !{i32 2, !"Debug Info Version", i32 3}
    "#};
    Test::init_subscriber();
    let caller = Location::caller();
    let (module, actual) = Test::opt(FLAGS, src);
    let module = module.try_read().unwrap();
    assert!(module.as_any().is::<targ3t::llvmir::ModuleOp>());
    Test::check_lines_contain(&actual, expected, caller);
}

#[test]
fn test_add_one() {
    let src = indoc! {"
    llvm.func @add_one(%arg0 : i32) -> i32 {
      %0 = llvm.mlir.constant(1 : i32) : i32
      %1 = llvm.add %0, %arg0 : i32
      llvm.return %1 : i32
    }
    "};
    let expected = indoc! {r#"
    define i32 @add_one(i32 %arg0) {
        %1 = add i32 %arg0, 1
        ret i32 %1
    }
    "#};
    Test::init_subscriber();
    let (_module, actual) = Test::parse(src);
    Test::check_lines_contain(&actual, &src, Location::caller());
    let (_module, actual) = Test::opt(FLAGS, src);
    Test::check_lines_contain(&actual, expected, Location::caller());
}
