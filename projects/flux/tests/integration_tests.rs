use flux_lang::{FluxCompiler, eval};

#[test]
fn test_simple_arithmetic() {
    let source = r#"
main: i32
main = 1 + 2 + 3
"#;
    let result = eval(source);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 6);
}

#[test]
fn test_function_call() {
    let source = r#"
inc: i32 -> i32
inc x = x + 1

main: i32
main = inc 41
"#;
    let result = eval(source);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 42);
}

#[test]
fn test_let_binding() {
    let source = r#"
main: i32
main =
  let x = 10 in
  let y = 20 in
  x + y
"#;
    let result = eval(source);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 30);
}

#[test]
fn test_if_expression() {
    let source = r#"
main: i32
main = if true then 42 else 0
"#;
    let result = eval(source);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 42);
}

#[test]
fn test_nested_if() {
    let source = r#"
main: i32
main = if false then 0 else if true then 99 else 1
"#;
    let result = eval(source);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 99);
}

#[test]
fn test_effect_check_pass() {
    let mut compiler = FluxCompiler::new();
    let source = r#"
foo: i32 -> i32 !{pure, cpu, alloc none}
foo x = x + 1

bar: i32 -> i32 !{pure, cpu, alloc none}
bar x = foo x

main: i32
main = bar 10
"#;
    let result = compiler.check_source(source);
    assert!(result.is_ok());
}

#[test]
fn test_effect_check_fail_allocation() {
    let mut compiler = FluxCompiler::new();
    let source = r#"
foo: i32 -> i32 !{pure, cpu, alloc heap}
foo x = x + 1

bar: i32 -> i32 !{pure, cpu, alloc none}
bar x = foo x

main: i32
main = bar 10
"#;
    let result = compiler.check_source(source);
    assert!(result.is_err());
}

#[test]
fn test_effect_check_fail_purity() {
    let mut compiler = FluxCompiler::new();
    let source = r#"
foo: i32 -> i32 !{io, cpu, alloc none}
foo x = x + 1

bar: i32 -> i32 !{pure, cpu, alloc none}
bar x = foo x

main: i32
main = bar 10
"#;
    let result = compiler.check_source(source);
    assert!(result.is_err());
}

#[test]
fn test_effect_check_concurrent() {
    let mut compiler = FluxCompiler::new();
    let source = r#"
foo: i32 -> i32 !{pure, cpu, alloc none, concurrent}
foo x = x * 2

bar: i32 -> i32 !{pure, cpu, alloc none, concurrent}
bar x = foo x

main: i32
main = bar 5
"#;
    let result = compiler.check_source(source);
    assert!(result.is_ok());
}

#[test]
fn test_default_effects() {
    let mut compiler = FluxCompiler::new();
    let source = r#"
inc: i32 -> i32
inc x = x + 1

main: i32
main = inc 1
"#;
    let result = compiler.check_source(source);
    assert!(result.is_ok());
}

#[test]
fn test_parse_error() {
    let compiler = FluxCompiler::new();
    let source = r#"
main: i32
main = 1 + + 2
"#;
    let result = compiler.parse(source);
    assert!(result.is_err());
}

#[test]
fn test_comparison_operators() {
    let source = r#"
main: i32
main = if 5 > 3 then 1 else 0
"#;
    let result = eval(source);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 1);
}

#[test]
fn test_logical_operators() {
    let source = r#"
main: i32
main = if true && false then 0 else 42
"#;
    let result = eval(source);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 42);
}

#[test]
fn test_negation() {
    let source = r#"
main: i32
main = -10
"#;
    let result = eval(source);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), -10);
}

#[test]
fn test_multiplication() {
    let source = r#"
main: i32
main = 6 * 7
"#;
    let result = eval(source);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 42);
}

#[test]
fn test_i8_literal_type_check() {
    let mut compiler = FluxCompiler::new();
    let source = r#"
main: i8
main = 42
"#;
    let result = compiler.check_source(source);
    assert!(result.is_ok());
}

#[test]
fn test_u8_literal_overflow() {
    let mut compiler = FluxCompiler::new();
    let source = r#"
main: u8
main = 300
"#;
    let result = compiler.check_source(source);
    assert!(result.is_err());
}


#[test]
fn test_division() {
    let source = r#"
main: i32
main = 84 / 2
"#;
    let result = eval(source);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 42);
}

#[test]
fn test_compile_to_wasm() {
    let mut compiler = FluxCompiler::new();
    let source = r#"
main: i32
main = 42
"#;
    let result = compiler.compile_source_wasm(source);
    assert!(result.is_ok());
    let wasm_bytes = result.unwrap();
    assert!(!wasm_bytes.is_empty());
}

#[test]
fn test_complex_expression() {
    let source = r#"
main: i32
main =
  let a = 2 in
  let b = 3 in
  let c = 4 in
  a * b + c
"#;
    let result = eval(source);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 10);
}

#[test]
fn test_nested_functions() {
    let source = r#"
double: i32 -> i32
double x = x * 2

quadruple: i32 -> i32
quadruple x = double (double x)

main: i32
main = quadruple 3
"#;
    let result = eval(source);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 12);
}
