extern crate xrcf;

use indoc::indoc;
use std::panic::Location;
use xrcf::tester::DefaultTester;

fn flags() -> Vec<&'static str> {
    vec!["--convert-func-to-llvm"]
}

#[test]
fn test_constant() {
    DefaultTester::init_tracing();
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
    let (module, actual) = DefaultTester::transform(flags(), src);
    DefaultTester::verify(module);
    DefaultTester::check_lines_contain(&actual, expected, Location::caller());
}

#[test]
fn test_add_one() {
    DefaultTester::init_tracing();
    let src = indoc! {"
    func.func @add_one(%arg0 : i32) -> i32 {
      %0 = arith.constant 1 : i32
      %1 = arith.addi %0, %arg0 : i32
      return %1 : i32
    }
    "};
    let expected = indoc! {"
    llvm.func @add_one(%arg0 : i32) -> i32 {
      %0 = llvm.mlir.constant(1 : i32) : i32
      %1 = llvm.add %0, %arg0 : i32
      llvm.return %1 : i32
    }
    "};
    let (module, actual) = DefaultTester::transform(flags(), src);
    DefaultTester::verify(module);
    DefaultTester::check_lines_contain(&actual, expected, Location::caller());
}

#[test]
fn test_hello_world() {
    DefaultTester::init_tracing();
    let src = indoc! {r#"
    func.func private @printf(!llvm.ptr) -> i32
    func.func @something() -> i32
    "#};
    let expected = indoc! {r#"
    llvm.func @printf(!llvm.ptr) -> i32 attributes {sym_visibility = "private"}
    llvm.func @something() -> i32
    "#};
    let (_module, actual) = DefaultTester::parse(src);
    DefaultTester::check_lines_contain(&actual, src, Location::caller());
    let (module, actual) = DefaultTester::transform(flags(), src);
    DefaultTester::verify(module);
    DefaultTester::check_lines_contain(&actual, expected, Location::caller());

    let src = indoc! {r#"
    func.func private @printf(!llvm.ptr) -> i32

    func.func @main() -> i32 {
      %0 = llvm.mlir.constant("hello, world\0A\00") : !llvm.array<14 x i8>

      %1 = arith.constant 14 : i64
      %2 = llvm.alloca %1 x i8 : (i64) -> !llvm.ptr

      llvm.store %0, %2 : !llvm.array<14 x i8>, !llvm.ptr
      %3 = func.call @printf(%2) : (!llvm.ptr) -> i32

      %4 = llvm.mlir.constant(0 : i32) : i32
      return %4 : i32
    }
    "#};
    let expected = indoc! {r#"
    llvm.func @printf(!llvm.ptr) -> i32 attributes {sym_visibility = "private"}

    llvm.func @main() -> i32 {
      %0 = llvm.mlir.constant("hello, world\0A\00") : !llvm.array<14 x i8>

      %1 = llvm.mlir.constant(14 : i64) : i64
      %2 = llvm.alloca %1 x i8 : (i64) -> !llvm.ptr

      llvm.store %0, %2 : !llvm.array<14 x i8>, !llvm.ptr
      %3 = llvm.call @printf(%2) : (!llvm.ptr) -> i32

      %4 = llvm.mlir.constant(0 : i32) : i32
      llvm.return %4 : i32
    }
    "#};
    let (_module, actual) = DefaultTester::parse(src);
    DefaultTester::check_lines_contain(&actual, src, Location::caller());
    let (module, actual) = DefaultTester::transform(flags(), src);
    DefaultTester::verify(module);
    DefaultTester::check_lines_contain(&actual, expected, Location::caller());
}

#[test]
fn test_empty_return() {
    DefaultTester::init_tracing();
    let src = indoc! {r#"
    func.func @main() {
      return
    }
    "#};
    let expected = indoc! {r#"
    module {
      llvm.func @main() {
        llvm.return
      }
    }
    "#};
    let (module, actual) = DefaultTester::transform(flags(), src);
    DefaultTester::verify(module);
    DefaultTester::check_lines_exact(&actual, expected, Location::caller());
}
