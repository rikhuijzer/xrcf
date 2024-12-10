extern crate xrcf;

use indoc::indoc;
use std::panic::Location;
use xrcf::tester::Tester;

fn flags() -> Vec<&'static str> {
    vec!["--convert-scf-to-cf"]
}

#[test]
fn test_if_with_yield() {
    Tester::init_tracing();
    let src = indoc! {r#"
    module {
      func.func @main() -> i32 {
        %0 = arith.constant false
        %1 = scf.if %0 -> (i32) {
          %0 = arith.constant 3 : i32
          scf.yield %0 : i32
        } else {
          %0 = arith.constant 4 : i32
          scf.yield %0 : i32
        }
        return %1 : i32
      }
    }
    "#};
    let expected = indoc! {r#"
    module {
      func.func @main() -> i32 {
        %0 = arith.constant false
        cf.cond_br %0, ^bb1, ^bb2
      ^bb1:
        %1 = arith.constant 3 : i32
        cf.br ^bb3(%1 : i32)
      ^bb2:
        %2 = arith.constant 4 : i32
        cf.br ^bb3(%2 : i32)
      ^bb3(%arg0 : i32):
        cf.br ^bb4
      ^bb4:
        return %arg0 : i32
      }
    }
    "#};
    let (module, actual) = Tester::parse(src);
    Tester::verify(module);
    Tester::check_lines_contain(&actual, &src, Location::caller());
    let (module, actual) = Tester::transform(flags(), src);
    Tester::verify(module);
    Tester::check_lines_exact(&actual, expected, Location::caller());
}

#[test]
fn test_if_without_yield() {
    Tester::init_tracing();
    let src = indoc! {r#"
    func.func @main() -> i64 {
      %0 = arith.constant false
      scf.if %0 {
        %0 = arith.constant 2 : i64
      } else {
        %0 = arith.constant 3 : i64
      }
      %1 = arith.constant 0 : i64
      return %1 : i64
    }
    "#};
    let expected = indoc! {r#"
    func.func @main() -> i64 {
      %0 = arith.constant false
      cf.cond_br %0, ^bb1, ^bb2
    ^bb1: 
      %1 = arith.constant 2 : i64
      cf.br ^bb3
    ^bb2:
      %2 = arith.constant 3 : i64
      cf.br ^bb3
    ^bb3:
      %3 = arith.constant 0 : i64
      return %3 : i64
    }
    "#};
    let (module, actual) = Tester::parse(src);
    Tester::verify(module);
    Tester::check_lines_contain(&actual, &src, Location::caller());
    let (module, actual) = Tester::transform(flags(), src);
    Tester::verify(module);
    Tester::check_lines_contain(&actual, expected, Location::caller());
}
