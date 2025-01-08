use indoc::indoc;
use xrcf::shared::SharedExt;
use xrcf::targ3t::clif::ModuleOp;
use xrcf::tester::DefaultTester;

fn flags() -> Vec<&'static str> {
    vec!["--convert-mlir-to-clif"]
}

#[test]
fn test_plus() {
    DefaultTester::init_tracing();
    let src = indoc! {r#"
    module {
        func.func @i32_plus(%arg0: i32, %arg1: i32) -> i32 {
            %0 = arith.addi %arg0, %arg1 : i32
            return %0 : i32
        }
    }
    "#};

    let (module, _actual) = DefaultTester::transform(flags(), src);
    DefaultTester::verify(module.clone());
    let bytes = module
        .rd()
        .as_any()
        .downcast_ref::<ModuleOp>()
        .unwrap()
        .machine_code()
        .unwrap();
    let bytes = unsafe { std::slice::from_raw_parts(bytes, 1024) };
    let (mut store, instance) = DefaultTester::load_wasm(bytes).unwrap();
    let plus = instance.get_func(&mut store, "i32_plus").unwrap();
    let plus = plus.typed::<(i32, i32), i32>(&store).unwrap();
    let result = plus.call(&mut store, (1, 2)).unwrap();
    assert_eq!(result, 3);
}
