extern crate xrcf;

use indoc::indoc;
use std::panic::Location;
use xrcf::tester::Tester;

fn flags() -> Vec<&'static str> {
    vec!["--convert-func-to-llvm"]
}

#[test]
fn test_constant() {
    Tester::init_tracing();
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
    let (module, actual) = Tester::transform(flags(), src);
    Tester::verify(module);
    Tester::check_lines_contain(&actual, expected, Location::caller());
}

#[test]
fn test_add_one() {
    Tester::init_tracing();
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
    let (module, actual) = Tester::transform(flags(), src);
    Tester::verify(module);
    Tester::check_lines_contain(&actual, expected, Location::caller());
}

#[test]
fn test_hello_world() {
    Tester::init_tracing();
    let src = indoc! {r#"
    func.func private @printf(!llvm.ptr) -> i32
    func.func @something() -> i32
    "#};
    let expected = indoc! {r#"
    llvm.func @printf(!llvm.ptr) -> i32 attributes {sym_visibility = "private"}
    llvm.func @something() -> i32
    "#};
    let (_module, actual) = Tester::parse(src);
    Tester::check_lines_contain(&actual, &src, Location::caller());
    let (module, actual) = Tester::transform(flags(), src);
    Tester::verify(module);
    Tester::check_lines_contain(&actual, expected, Location::caller());

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
    let (_module, actual) = Tester::parse(src);
    Tester::check_lines_contain(&actual, &src, Location::caller());
    let (module, actual) = Tester::transform(flags(), src);
    Tester::verify(module);
    Tester::check_lines_contain(&actual, expected, Location::caller());
}

#[test]
fn test_empty_return() {
    Tester::init_tracing();
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
    let (module, actual) = Tester::transform(flags(), src);
    Tester::verify(module);
    Tester::check_lines_exact(&actual, expected, Location::caller());
}

#[test]
fn test_if_else() {
    Tester::init_tracing();
    let src = indoc! {r#"
    module {
      func.func @main() -> i32 {
        %false = arith.constant false
        cf.cond_br %false, ^bb1, ^bb2
      ^bb1:  // pred: ^bb0
        %c3_i32 = arith.constant 3 : i32
        cf.br ^bb3(%c3_i32 : i32)
      ^bb2:  // pred: ^bb0
        %c4_i32 = arith.constant 4 : i32
        cf.br ^bb3(%c4_i32 : i32)
      ^bb3(%0: i32):  // 2 preds: ^bb1, ^bb2
        cf.br ^bb4
      ^bb4:  // pred: ^bb3
        return %0 : i32
      }
    }
    "#};
    let _expected = indoc! {r#"
    module {
      llvm.func @main() -> i32 {
        %0 = llvm.mlir.constant(false) : i1
        llvm.cond_br %0, ^bb1, ^bb2
      ^bb1:  // pred: ^bb0
        %1 = llvm.mlir.constant(3 : i32) : i32
        llvm.br ^bb3(%1 : i32)
      ^bb2:  // pred: ^bb0
        %2 = llvm.mlir.constant(4 : i32) : i32
        llvm.br ^bb3(%2 : i32)
      ^bb3(%3: i32):  // 2 preds: ^bb1, ^bb2
        llvm.br ^bb4
      ^bb4:  // pred: ^bb3
        llvm.return %3 : i32
      }
    }
    "#};
    let (_module, actual) = Tester::parse(src);
    Tester::check_lines_exact(&actual, &src, Location::caller());
}