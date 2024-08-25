; Returns 3 when executed via `lli hello-world.ll` on LLVM 18.1.8.
@three = private constant i32 3

define i32 @main() {
entry:
  %loaded = load i32, i32* @one
  ret i32 %loaded
}
