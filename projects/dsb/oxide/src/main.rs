pub mod ast;
pub mod codegen;
pub mod lexer;
pub mod parser;
pub mod typechecker;

use ast::Expr;

fn main() {
    println!("--- Oxide Compiler (Function Application & Operators/Lambda Test) ---");

    let input = "map f (filter p xs)";
    println!("Input: \"{}\"", input);

    // 1. Lexing
    let tokens = lexer::lex(input).expect("Lexing failed");
    println!("Lexer output: {:?}", tokens);

    // 2. Parsing
    let mut parser = parser::Parser::new(tokens);
    let ast = parser.parse().expect("Parsing failed");
    println!("Parser output: {:#?}", ast);

    // 3. Verification
    // We expect the AST to be `App(App(map, f), App(App(filter, p), xs))`
    let expected_ast = Expr::App {
        func: Box::new(Expr::App {
            func: Box::new(Expr::Variable("map".to_string())),
            args: vec![Expr::Variable("f".to_string())],
        }),
        args: vec![Expr::App {
            func: Box::new(Expr::App {
                func: Box::new(Expr::Variable("filter".to_string())),
                args: vec![Expr::Variable("p".to_string())],
            }),
            args: vec![Expr::Variable("xs".to_string())],
        }],
    };

    assert_eq!(ast, expected_ast);
    println!("\nAST correctly represents function application.");

    // Additional tests
    // 1) Binary operator precedence: 1 + 2 * 3 -> (1 + (2 * 3))
    let input2 = "1 + 2 * 3";
    let tokens2 = lexer::lex(input2).expect("Lexing failed");
    let mut parser2 = parser::Parser::new(tokens2);
    let ast2 = parser2.parse().expect("Parsing failed");
    use ast::{Literal::*, Op::*};
    let expected2 = Expr::BinaryOp(
        Box::new(Expr::Literal(Nat(1))),
        Add,
        Box::new(Expr::BinaryOp(
            Box::new(Expr::Literal(Nat(2))),
            Mul,
            Box::new(Expr::Literal(Nat(3))),
        )),
    );
    assert_eq!(ast2, expected2);
    println!("Binary precedence OK: {}", input2);

    // 2) Lambda parsing: |x y| x + y
    let input3 = "|x y| x + y";
    let tokens3 = lexer::lex(input3).expect("Lexing failed");
    let mut parser3 = parser::Parser::new(tokens3);
    let ast3 = parser3.parse().expect("Parsing failed");
    let expected3 = Expr::Lambda {
        params: vec!["x".to_string(), "y".to_string()],
        body: Box::new(Expr::BinaryOp(
            Box::new(Expr::Variable("x".to_string())),
            Add,
            Box::new(Expr::Variable("y".to_string())),
        )),
    };
    assert_eq!(ast3, expected3);
    println!("Lambda parsing OK: {}", input3);

    // 3) Move operator: #x + 1 -> BinaryOp(Move(Variable x), Add, Literal 1)
    let input4 = "#x + 1";
    let tokens4 = lexer::lex(input4).expect("Lexing failed");
    let mut parser4 = parser::Parser::new(tokens4);
    let ast4 = parser4.parse().expect("Parsing failed");
    let expected4 = Expr::BinaryOp(
        Box::new(Expr::Move(Box::new(Expr::Variable("x".to_string())))),
        Add,
        Box::new(Expr::Literal(Nat(1))),
    );
    assert_eq!(ast4, expected4);
    println!("Move operator OK: {}", input4);
}
