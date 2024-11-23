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
    let (module, actual) = Tester::transform(flags(), src);
    Tester::verify(module.clone());
    let module = module.try_read().unwrap();
    assert!(module.as_any().is::<targ3t::llvmir::ModuleOp>());
    Tester::check_lines_contain(&actual, expected, Location::caller());
}

#[test]
fn test_empty_return() {
    Tester::init_tracing();
    let src = indoc! {r#"
    llvm.func @main() {
      llvm.return
    }
    "#};
    let expected = indoc! {"
    define void @main() {
      ret void
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
    llvm.func @add_one(%arg0 : i32) -> i32 {
      %0 = llvm.mlir.constant(1 : i32) : i32
      %1 = llvm.add %0, %arg0 : i32
      llvm.return %1 : i32
    }
    "};
    let expected = indoc! {r#"
    define i32 @add_one(i32 %arg0) {
        %1 = add i32 1, %arg0
        ret i32 %1
    }
    "#};
    let (_module, actual) = Tester::parse(src);
    Tester::check_lines_contain(&actual, &src, Location::caller());
    let (module, actual) = Tester::transform(flags(), src);
    Tester::verify(module);
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
    let (module, actual) = Tester::transform(flags(), src);
    Tester::verify(module);
    Tester::check_lines_contain(&actual, expected, Location::caller());
}

#[test]
fn test_print_with_vararg() {
    Tester::init_tracing();
    let src = indoc! {r#"
    llvm.func @printf(!llvm.ptr, ...) -> i32 attributes {sym_visibility = "private"}

    llvm.func @main() -> i32 {
      %0 = llvm.mlir.constant("hello, %d\0A\00") : !llvm.array<11 x i8>
      %1 = llvm.mlir.constant(14 : i64) : i64
      %2 = llvm.alloca %1 x i8 : (i64) -> !llvm.ptr
      llvm.store %0, %2 : !llvm.array<11 x i8>, !llvm.ptr
      %3 = llvm.mlir.constant(42 : i32) : i32
      %4 = llvm.call @printf(%2, %3) vararg(!llvm.func<i32 (ptr, ...)>) :
        (!llvm.ptr, i32) -> i32
      %5 = llvm.mlir.constant(0 : i32) : i32
      llvm.return %5 : i32
    }
    "#};
    let expected = indoc! {r#"
    ; ModuleID = 'LLVMDialectModule'
    source_filename = "LLVMDialectModule"

    declare i32 @printf(ptr, ...)
    define i32 @main() {
      %2 = alloca i8, i64 14, align 1
      store [11 x i8] c"hello, %d\0A\00", ptr %2, align 1
      %4 = call i32 (ptr, ...) @printf(ptr %2, i32 42)
      ret i32 0
    }

    !llvm.module.flags = !{!0}

    !0 = !{i32 2, !"Debug Info Version", i32 3}
    "#};
    let (module, actual) = Tester::transform(flags(), src);
    Tester::verify(module);
    Tester::check_lines_exact(&actual, expected, Location::caller());
}

#[test]
fn test_call_hello() {
    Tester::init_tracing();
    let src = indoc! {r#"
    llvm.func @hello() {
      llvm.return
    }
    llvm.func @main() {
      llvm.call @hello() : () -> ()
      llvm.return
    }
    "#};
    let expected = indoc! {r#"
    define void @hello() {
      ret void
    }
    define void @main() {
      call void @hello()
      ret void
    }
    "#};
    let (module, actual) = Tester::transform(flags(), src);
    Tester::verify(module);
    Tester::check_lines_contain(&actual, expected, Location::caller());
}

#[test]
fn test_if_else() {
    Tester::init_tracing();
    let src = indoc! {r#"
    module {
      llvm.func @main() -> i32 {
        %false = llvm.mlir.constant(false) : i1
        llvm.cond_br %false, ^then, ^else
      ^then:
        %c3_i32 = llvm.mlir.constant(3 : i32) : i32
        llvm.br ^merge(%c3_i32 : i32)
      ^else:
        %c4_i32 = llvm.mlir.constant(4 : i32) : i32
        llvm.br ^merge(%c4_i32 : i32)
      ^merge(%result : i32):
        llvm.br ^exit
      ^exit:
        llvm.return %result : i32
      }
    }
    "#};
    let expected = indoc! {r#"
    ; ModuleID = 'LLVMDialectModule'
    source_filename = "LLVMDialectModule"

    define i32 @main() {
      br i1 false, label %then, label %else
    then:
      br label %merge
    else:
      br label %merge
    merge:
      %4 = phi i32 [ 4, %else ], [ 3, %then ]
      br label %exit
    exit:
      ret i32 %4
    }

    !llvm.module.flags = !{!0}

    !0 = !{i32 2, !"Debug Info Version", i32 3}
    "#};
    let (module, actual) = Tester::transform(flags(), src);
    Tester::verify(module);
    Tester::check_lines_exact(&actual, expected, Location::caller());
}
