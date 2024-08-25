extern crate rrcf;

use rrcf::translate;

#[test]
fn test_translate() {
    let mlir = "llvm.mlir.global internal @i32_global(42: i32) : i32";
    let llvmir = "i32_global = internal global i32 42";
    
    assert_eq!(translate(mlir), llvmir);
}
