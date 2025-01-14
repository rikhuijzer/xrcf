# wea

This directory contains an example compiler built using xrcf that can compile and run a small program in the experimental wea language.

## Design

wea is meant to create kernels (core logic) that compile directly to WebAssembly.
Like C, it's meant to have a lot of control over the underlying assembly.

```wea
fn plus(a: i32, b: i32) i32 {
    a + b
}
```

Apart from the missing `->`, the syntax is very similar to Rust.

The language uses curly braces for blocks.
As Chris Lattner has pointed out during a podcast, this is indeed a bit redundant since most programmers will also use indentation anyway.
However, when removing the curly braces, it can sometimes be very difficult for readers to see where a block ends (thanks to Jose Storopoli for pointing this out).
For example, C's `if` statement may omit the curly braces:

```c
int do_something(user, password) {
    if is_valid(password)
        store(username)
        authorize(username)
}
```

Here, `authorize` will not be called if `is_valid` returns true because only the first line after the `if` statement is executed.
Formatters would detect this of course, but without them it can be very tricky.
