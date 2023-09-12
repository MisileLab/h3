# eXSL and eXSL Library specification v0.1

## Introduction

This document describes the eXSL and eXSL Library specification.

## Why eXSL?

Other language like Rust, Go, C++ and more lang needs to wrapping DOM for webassembly.\
So, I made this language, for wasm, first target is wasm, official DOM wrapper.\
This is eXSL.

## eXSL Language Specification

### Basic Syntax

```exsl
a {
    b()
    c()
}
```

no semicolon, yes newline, yes brace

### Variables

```exsl
a = 0
mut a = 0
var a = 0
```

normal variable is immutable value and immutable inner value\
mut variable is immutable value and mutable inner value\
var variable is mutable value and mutable inner value

### Function

```exsl
func a(b: u64, c: u64, d: u64) -> u64 {
    ret b+c+d
}
```

### Statements

```exsl
+,-,*,/
```

### Types

```exsl
u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, f32, f64
string
```

## eXSL Library Specification

### Functions

```exsl
print(string)
input()
```
