extern crate xrcf;

use crate::tester::Test;
use indoc::indoc;
use std::panic::Location;
use xrcf::ir::ModuleOp;

fn flags() -> Vec<&'static str> {
    vec!["--convert-unstable-to-mlir"]
}

#[test]
fn test_constant() {
    // Note that `\n` is escaped to `\\n` by `indoc!`.
    let src = indoc! {r#"
    func.func @main() -> i32 {
      %0 = arith.constant 0 : i32
      unstable.printf("hello, world\n")
      return %0 : i32
    }
    "#};

    let expected = indoc! {r#"
    func.func private @printf(!llvm.ptr) -> i32

    func.func @main() -> i32 {
      %0 = arith.constant 0 : i32
      %1 = llvm.mlir.constant("hello, world\n\00") : !llvm.array<14 x i8>
      %2 = arith.constant 14 : i16
      %3 = llvm.alloca %2 x i8 : (i16) -> !llvm.ptr
      llvm.store %1, %3 : !llvm.array<14 x i8>, !llvm.ptr
      %4 = func.call @printf(%3) : (!llvm.ptr) -> i32
      return %0 : i32
    }
    "#};
    Test::init_tracing();
    let caller = Location::caller();
    let (module, actual) = Test::transform(flags(), src);
    let module = module.try_read().unwrap();
    assert!(module.as_any().is::<ModuleOp>());
    Test::check_lines_contain(&actual, expected, caller);
}

#[test]
fn test_two_constants() {
    // Note that `\n` is escaped to `\\n` by `indoc!`.
    let src = indoc! {r#"
    func.func @main() -> i32 {
      %0 = arith.constant 0 : i32
      unstable.printf("hello, world\n")
      unstable.printf("hello 2\n")
      return %0 : i32
    }
    "#};
    Test::init_tracing();
    let (_module, actual) = Test::transform(flags(), src);
    assert_eq!(actual.matches("func.func private @printf").count(), 1);
}
