extern crate xrcf;

use indoc::indoc;
use std::panic::Location;
use xrcf::targ3t;
use xrcf::tester::Tester;

fn flags() -> Vec<&'static str> {
    vec!["--convert-mlir-to-llvmir"]
}

#[test]
fn test_constant() {
    Tester::init_tracing();
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
    let caller = Location::caller();
    let (module, actual) = Tester::transform(flags(), src);
    let module = module.try_read().unwrap();
    assert!(module.as_any().is::<targ3t::llvmir::ModuleOp>());
    Tester::check_lines_contain(&actual, expected, caller);
}

#[test]
fn test_add_one() {
    Tester::init_tracing();
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
    let (_module, actual) = Tester::parse(src);
    Tester::check_lines_contain(&actual, &src, Location::caller());
    let (_module, actual) = Tester::transform(flags(), src);
    Tester::check_lines_contain(&actual, expected, Location::caller());
}

#[test]
fn test_hello_world() {
    Tester::init_tracing();
    let src = indoc! {r#"
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
    let expected = indoc! {r#"
    ; ModuleID = 'LLVMDialectModule'
    source_filename = "LLVMDialectModule"

    declare i32 @printf(ptr)

    define i32 @main() {
      %2 = alloca i8, i64 14, align 1
      store [14 x i8] c"hello, world\0A\00", ptr %2, align 1
      %3 = call i32 @printf(ptr %2)
      ret i32 0
    }

    !llvm.module.flags = !{!0}

    !0 = !{i32 2, !"Debug Info Version", i32 3}
    "#};
    let (_module, actual) = Tester::parse(src);
    Tester::check_lines_contain(&actual, &src, Location::caller());
    let (_module, actual) = Tester::transform(flags(), src);
    Tester::check_lines_contain(&actual, expected, Location::caller());
}
