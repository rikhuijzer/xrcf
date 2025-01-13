extern crate xrcf;

use indoc::indoc;
use std::panic::Location;
use xrcf::tester::DefaultTester;

fn flags() -> Vec<&'static str> {
    vec!["--convert-cf-to-llvm"]
}

#[test]
fn test_if_else() {
    DefaultTester::init_tracing();
    let src = indoc! {r#"
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
    let expected = indoc! {r#"
    module {
        func.func @main() -> i32 {
            %0 = arith.constant false
            llvm.cond_br %0, ^bb1, ^bb2
        ^bb1:
            %1 = arith.constant 3 : i32
            llvm.br ^bb3(%1 : i32)
        ^bb2:
            %2 = arith.constant 4 : i32
            llvm.br ^bb3(%2 : i32)
        ^bb3(%arg0 : i32):
            llvm.br ^bb4
        ^bb4:
            return %arg0 : i32
        }
    }
    "#};
    let (module, actual) = DefaultTester::parse(src);
    DefaultTester::check_lines_exact(&actual, src, Location::caller());
    DefaultTester::verify(module);
    let (module, actual) = DefaultTester::transform(flags(), src);
    DefaultTester::check_lines_exact(&actual, expected, Location::caller());
    DefaultTester::verify(module);
}
