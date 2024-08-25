; Returns 3 when executed via `lli hello-world.ll` on LLVM 18.1.8.
@one = private constant i32 1
@two = private constant i32 2

define i32 @main() {
entry:
  %loadedone = load i32, i32* @one
  %loadedtwo = load i32, i32* @two
  %out = add i32 %loadedone, %loadedtwo
  ret i32 %out
}