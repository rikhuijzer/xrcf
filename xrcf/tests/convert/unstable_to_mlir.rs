extern crate xrcf;

use indoc::indoc;
use std::panic::Location;
use xrcf::ir::ModuleOp;
use xrcf::tester::Tester;

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
    Tester::init_tracing();
    let caller = Location::caller();
    let (module, actual) = Tester::transform(flags(), src);
    Tester::verify(module.clone());
    let module = module.try_read().unwrap();
    assert!(module.as_any().is::<ModuleOp>());
    Tester::check_lines_contain(&actual, expected, caller);
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
    Tester::init_tracing();
    let (module, actual) = Tester::transform(flags(), src);
    Tester::verify(module);
    assert_eq!(actual.matches("func.func private @printf").count(), 1);
}

#[test]
fn test_hello_world() {
    let src = indoc! {r#"
    module {
      func.func @hello() {
        unstable.printf("Hello, World!")
        return
      }
      func.func @main() -> i32 {
        %0 = arith.constant 0 : i32
        func.call @hello() : () -> ()
        return %0 : i32
      }
    }
    "#}
    .trim();
    let expected = indoc! {r#"
    module {
      func.func private @printf(!llvm.ptr) -> i32
      func.func @hello() {
        %0 = llvm.mlir.constant("Hello, World!\00") : !llvm.array<14 x i8>
        %1 = arith.constant 14 : i16
        %2 = llvm.alloca %1 x i8 : (i16) -> !llvm.ptr
        llvm.store %0, %2 : !llvm.array<14 x i8>, !llvm.ptr
        %3 = func.call @printf(%2) : (!llvm.ptr) -> i32
        return
      }
      func.func @main() -> i32 {
        %0 = arith.constant 0 : i32
        func.call @hello() : () -> ()
        return %0 : i32
      }
    }
    "#};
    Tester::init_tracing();
    let (_module, actual) = Tester::transform(flags(), src);
    Tester::check_lines_exact(&actual, expected, Location::caller());
}
