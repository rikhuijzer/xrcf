; Prints hello world when executed via `lli hello-world.ll` on LLVM 18.1.8.
@text = private constant [15 x i8] c"hello, world!\0A\00"

define i32 @main() {
entry:
  %str = getelementptr inbounds [15 x i8], [15 x i8]* @text, i32 0, i32 0
  %call = call i32 (i8*, ...)* @printf(i8* %str)
  ret i32 1
}

declare i32 @printf(i8*, ...)
