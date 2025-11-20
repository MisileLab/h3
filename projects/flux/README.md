# Flux Programming Language

Flux is an experimental functional programming language with a sophisticated effect system, zero-allocation guarantees, and native support for parallel computing.

## Features

### Core Language Features
- **Haskell-style syntax** with type signatures and pattern matching
- **Effect system** tracking purity, execution space, allocation, and concurrency
- **Zero-allocation by default** - functions are `alloc none` unless explicitly declared otherwise
- **Arena-based memory management** for predictable, region-based allocation
- **First-class parallel primitives** - `par_for`, `par_map`, `par_map_inplace`
- **Async/await** for structured concurrency with Task types
- **Actor model** for message-passing concurrency
- **GPU computation support** with CPU/GPU execution space tracking
- **Unsafe blocks** for low-level system programming

### Effect System

Functions declare their effects explicitly:

```flux
foo: i32 -> i32 !{pure, cpu, alloc none}
foo x = x + 1
```

Effect dimensions:
- **Purity**: `pure`, `io`, `state`, `debug`
- **Execution**: `cpu`, `gpu`
- **Allocation**: `alloc none`, `alloc arena`, `alloc heap`
- **Concurrency**: `single`, `concurrent`

The compiler enforces that callers have sufficient effects to call callees (superset rule).

### Zero-Allocation Guarantees

Functions marked with `alloc none` (the default) cannot:
- Allocate on the heap
- Create new arenas
- Call functions that perform allocation

This is verified at compile-time, preventing accidental allocations in performance-critical code.

## Project Structure

```
flux_lang/
├── Cargo.toml                    # Project manifest
├── src/
│   ├── main.rs                   # CLI tool (fluxc)
│   ├── lib.rs                    # Library API
│   ├── ast.rs                    # Abstract syntax tree definitions
│   ├── lexer.rs                  # Tokenizer
│   ├── parser.rs                 # Recursive descent parser
│   ├── types.rs                  # Type system with unification
│   ├── effect.rs                 # Effect checking and inference
│   ├── typecheck.rs              # Type and effect checker
│   ├── ir.rs                     # Intermediate representation
│   ├── codegen.rs                # IR generation from AST
│   ├── backend_x86.rs            # x86-64 backend (interpreter)
│   ├── backend_wasm.rs           # WebAssembly backend
│   ├── runtime.rs                # Runtime primitives (arena, parallel, async)
│   └── actor.rs                  # Actor system implementation
├── examples/                     # Example Flux programs
│   ├── basic.flux                # Simple pure functions
│   ├── zero_alloc_violation.flux # Effect violation example
│   ├── gpu_kernel.flux           # GPU computation
│   ├── parallel.flux             # Parallel primitives
│   ├── async_tasks.flux          # Async/await
│   └── actors.flux               # Actor-based concurrency
└── tests/                        # Integration tests
    └── integration_tests.rs
```

## Building

```bash
cargo build
```

## Usage

### Type-check a program

```bash
cargo run -- check examples/basic.flux
```

### Compile to WebAssembly

```bash
cargo run -- build examples/basic.flux --target wasm32 -o output.wasm
```

### Compile and run

```bash
cargo run -- run examples/basic.flux
```

## Example Programs

### Simple arithmetic with effect inference

```flux
inc: i32 -> i32
inc x = x + 1

main: i32
main = inc 41
```

### Effect violation detection

```flux
foo: i32 -> i32 !{pure, cpu, alloc heap}
foo x = x + 1

bar: i32 -> i32 !{pure, cpu, alloc none}
bar x = foo x  -- ERROR: Cannot call alloc heap from alloc none context
```

### Parallel computation

```flux
process: i32 -> i32 !{pure, cpu, alloc none, concurrent}
process x = x * 2

main: i32
main = process 21
```

## Implementation Details

### Compilation Pipeline

1. **Lexing** - Source code → Tokens
2. **Parsing** - Tokens → AST
3. **Type Checking** - AST → Typed AST + Effect validation
4. **IR Generation** - AST → Common IR
5. **Backend Compilation**:
   - x86-64: IR → Interpreted execution (JIT compilation would require LLVM/Cranelift)
   - WebAssembly: IR → WASM bytecode

### Effect Checking Algorithm

The effect checker validates that:
1. Function bodies only use effects declared in their signature
2. Function calls satisfy the superset rule: `caller_effects ⊇ callee_effects`
3. Special rules for each effect dimension (e.g., pure cannot call io)

### Memory Model

- **Arena allocator**: Fast, region-based allocation with bulk deallocation
- **Array types**: Track both execution space (CPU/GPU) and owning arena
- **Zero-copy semantics** for array operations where possible

### Runtime

- **Parallel primitives**: Built on Rayon for work-stealing parallelism
- **Async runtime**: Built on Tokio for cooperative multitasking
- **Actor system**: Message-passing with isolated state using crossbeam channels

## Current Limitations

- Parser requires function signatures and bodies on the same line
- Limited pattern matching support
- GPU backend is stubbed (no actual GPU code generation)
- No garbage collector (relies on arena/manual memory management)
- Limited standard library

## Future Work

- Full native code generation via LLVM or Cranelift
- Actual GPU code generation (CUDA/SPIR-V/Metal)
- Incremental compilation
- Module system
- Comprehensive standard library
- IDE support (LSP)
- Debugger integration

## Testing

Run all tests:

```bash
cargo test
```

Run specific test suite:

```bash
cargo test --test integration_tests
cargo test --lib
```

## Architecture Decisions

### Why an interpreter for x86-64?

The current x86-64 backend uses interpretation rather than native code generation to avoid dependencies on LLVM (which requires system installation) or Cranelift (which added significant compile times). A production implementation would use proper JIT compilation.

### Why separate effect dimensions?

Separating purity, allocation, execution space, and concurrency allows fine-grained control over function behavior while keeping the type system tractable. Each dimension can be reasoned about independently.

### Why default to zero allocation?

Making `alloc none` the default encourages developers to think about allocation upfront and makes performance characteristics explicit in the type system.

## License

This is an experimental language implementation created for educational purposes.

## Contributing

This is a prototype implementation. Contributions welcome!

Key areas for improvement:
- Better error messages with source locations
- Support for multi-line function declarations in parser
- Native code generation backend
- Standard library development
- Documentation and tutorials
