extern crate xrcf;

use indoc::indoc;
use std::panic::Location;
use xrcf::tester::Tester;

fn flags() -> Vec<&'static str> {
    vec!["--convert-cf-to-llvm"]
}

#[test]
fn test_if_else() {
    Tester::init_tracing();
    let src = indoc! {r#"
    module {
      func.func @main() -> i32 {
        %0 = arith.constant false
        cf.cond_br %0, ^then, ^else
      ^then:
        %1 = arith.constant 3 : i32
        cf.br ^merge(%1 : i32)
      ^else:
        %2 = arith.constant 4 : i32
        cf.br ^merge(%2 : i32)
      ^merge(%arg0 : i32):
        cf.br ^exit
      ^exit:
        return %arg0 : i32
      }
    }
    "#};
    let expected = indoc! {r#"
    module {
      func.func @main() -> i32 {
        %0 = arith.constant false
        llvm.cond_br %0, ^then, ^else
      ^then:
        %1 = arith.constant 3 : i32
        llvm.br ^merge(%1 : i32)
      ^else:
        %2 = arith.constant 4 : i32
        llvm.br ^merge(%2 : i32)
      ^merge(%arg0 : i32):
        llvm.br ^exit
      ^exit:
        return %arg0 : i32
      }
    }
    "#};
    let (module, actual) = Tester::parse(src);
    Tester::check_lines_exact(&actual, &src, Location::caller());
    Tester::verify(module);
    let (module, actual) = Tester::transform(flags(), src);
    Tester::check_lines_exact(&actual, expected, Location::caller());
    Tester::verify(module);
}
