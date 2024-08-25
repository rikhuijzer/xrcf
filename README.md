# rrcf

RRCF Compiler Framework

## But you don't know anything about compilers?

Yet.

## Notes

- Let's not include any LLVM code like Zig is moving towards.
  LLVM is great, but it's also big.
  Including the C files would require lots of compilation.
  Calling out to an installed LLVM version is better but is hard with package versions and such.
  Let's for now just generate LLVM IR and let the client compile it themselves.
  