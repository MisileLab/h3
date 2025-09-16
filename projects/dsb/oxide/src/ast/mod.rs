
//! Abstract Syntax Tree (AST) for the Oxide language.

/// A top-level item in a program.
#[derive(Debug, Clone, PartialEq)]
pub enum TopLevel {
    FunctionDef(FunctionDef),
    // Later: data, cls, inst, etc.
}

/// A function definition.
/// e.g., `process : @List Nat -> List Nat = { process @xs = ... }`
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDef {
    pub name: String,
    pub signature: Option<Type>,
    pub body: Expr,
}

/// A type expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    /// A named type, like `Nat` or `String`.
    TypeName(String),
    /// A type variable, like `a`.
    TypeVar(String),
    /// A function type, like `A -> B -> C`.
    Function {
        params: Vec<Type>,
        ret: Box<Type>,
    },
    /// A generic type application, like `List Nat` or `Array Float`.
    Generic(String, Vec<Type>),
    /// A borrowed type, like `@List Nat`.
    Borrow(Box<Type>),
}

/// An expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A literal value.
    Literal(Literal),
    /// A variable identifier.
    Variable(String),
    /// A binary operation, like `x + y`.
    BinaryOp(Box<Expr>, Op, Box<Expr>),
    /// A function call, like `f x y`.
    App {
        func: Box<Expr>,
        args: Vec<Expr>,
    },
    /// A lambda expression, like `|x| x + 1`.
    Lambda {
        params: Vec<String>,
        body: Box<Expr>,
    },
    /// A block of code, like `{ ... }`.
    Block(Vec<Expr>),
    /// An explicit move, like `#var`.
    Move(Box<Expr>),
}

/// A literal value.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Nat(u64),
    Float(f64),
    String(String),
    Bool(bool),
}

/// A binary operator.
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
}
