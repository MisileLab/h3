use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use flux_lang::{FluxCompiler, eval};
use std::fs;

mod rust_impls;

fn read_flux_program(name: &str) -> String {
    fs::read_to_string(format!("benches/programs/{}.flux", name))
        .expect("Failed to read flux program")
}

// Benchmark: Compilation phases
fn bench_compilation_phases(c: &mut Criterion) {
    let mut group = c.benchmark_group("compilation_phases");

    let source = read_flux_program("fibonacci");

    group.bench_function("lexer", |b| {
        b.iter(|| {
            let mut lexer = flux_lang::lexer::Lexer::new(black_box(&source));
            lexer.tokenize()
        })
    });

    group.bench_function("parser", |b| {
        b.iter(|| {
            let mut parser = flux_lang::parser::Parser::new(black_box(&source));
            parser.parse_module()
        })
    });

    group.bench_function("typecheck", |b| {
        let mut compiler = FluxCompiler::new();
        b.iter(|| {
            compiler.check_source(black_box(&source))
        })
    });

    group.bench_function("full_compile", |b| {
        let mut compiler = FluxCompiler::new();
        b.iter(|| {
            compiler.compile_source_x86(black_box(&source))
        })
    });

    group.finish();
}

// Benchmark: Flux execution vs Rust native
fn bench_fibonacci(c: &mut Criterion) {
    let mut group = c.benchmark_group("fibonacci");

    let flux_source = read_flux_program("fibonacci");

    group.bench_function("flux_fib_15", |b| {
        b.iter(|| {
            eval(black_box(&flux_source)).unwrap()
        })
    });

    group.bench_function("rust_fib_15", |b| {
        b.iter(|| {
            rust_impls::fibonacci(black_box(15))
        })
    });

    group.finish();
}

fn bench_factorial(c: &mut Criterion) {
    let mut group = c.benchmark_group("factorial");

    let flux_source = read_flux_program("factorial");

    group.bench_function("flux_fact_15", |b| {
        b.iter(|| {
            eval(black_box(&flux_source)).unwrap()
        })
    });

    group.bench_function("rust_fact_15", |b| {
        b.iter(|| {
            rust_impls::factorial(black_box(15))
        })
    });

    group.finish();
}

fn bench_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum");

    let flux_source = read_flux_program("sum");

    group.bench_function("flux_sum_1000", |b| {
        b.iter(|| {
            eval(black_box(&flux_source)).unwrap()
        })
    });

    group.bench_function("rust_sum_1000", |b| {
        b.iter(|| {
            rust_impls::sum(black_box(1000))
        })
    });

    group.finish();
}

fn bench_ackermann(c: &mut Criterion) {
    let mut group = c.benchmark_group("ackermann");

    let flux_source = read_flux_program("ackermann");

    group.bench_function("flux_ack_3_6", |b| {
        b.iter(|| {
            eval(black_box(&flux_source)).unwrap()
        })
    });

    group.bench_function("rust_ack_3_6", |b| {
        b.iter(|| {
            rust_impls::ackermann(black_box(3), black_box(6))
        })
    });

    group.finish();
}

// Benchmark: Compiler throughput with different program sizes
fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");

    for size in [10, 50, 100].iter() {
        // Generate a program with `size` nested function calls
        let program = format!(
            "add: i32 -> i32\nadd x = x + 1\n\n{}main: i32\nmain = {}",
            (0..*size).map(|i| format!("f{}: i32 -> i32\nf{} x = add x\n\n", i, i)).collect::<String>(),
            (0..*size).rev().fold("42".to_string(), |acc, i| format!("f{} ({})", i, acc))
        );

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_functions", size)),
            &program,
            |b, prog| {
                b.iter(|| {
                    eval(black_box(prog)).unwrap()
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_compilation_phases,
    bench_fibonacci,
    bench_factorial,
    bench_sum,
    bench_ackermann,
    bench_throughput
);

criterion_main!(benches);
