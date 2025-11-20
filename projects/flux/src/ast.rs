
#[derive(Debug, Clone, PartialEq)]
pub struct Module {
    pub declarations: Vec<Declaration>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Declaration {
    Function(FunctionDecl),
    TypeAlias(TypeAliasDecl),
    DataType(DataTypeDecl),
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDecl {
    pub name: String,
    pub type_sig: Option<TypeSignature>,
    pub params: Vec<String>,
    pub body: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeSignature {
    pub param_types: Vec<Type>,
    pub return_type: Box<Type>,
    pub effects: Option<EffectSet>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeAliasDecl {
    pub name: String,
    pub type_params: Vec<String>,
    pub aliased_type: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DataTypeDecl {
    pub name: String,
    pub type_params: Vec<String>,
    pub constructors: Vec<Constructor>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Constructor {
    pub name: String,
    pub fields: Vec<Type>,
}

pub type LiteralId = usize;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    Bool,
    Unit,
    String,
    Char,

    // Special type for integer literals before they are resolved
    IntLiteral(LiteralId),

    // Function type: (params, return, effects)
    Function(Vec<Type>, Box<Type>, EffectSet),

    // Type variable
    Var(String),

    // Type application
    App(String, Vec<Type>),

    // Array type: Array<T, Space, Arena>
    Array(Box<Type>, Space, ArenaRef),

    // Arena reference
    Arena,

    // Task type for async
    Task(Box<Type>),

    // Actor type
    Actor(Box<Type>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntegerKind {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Space {
    Cpu,
    Gpu,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ArenaRef {
    Named(String),
    Anonymous,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct EffectSet {
    pub purity: Purity,
    pub execution: Execution,
    pub allocation: Allocation,
    pub concurrency: Concurrency,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Purity {
    Pure,
    Io,
    State,
    Debug,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Execution {
    Cpu,
    Gpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Allocation {
    None,
    Arena,
    Heap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Concurrency {
    Single,
    Concurrent,
}

impl Default for Purity {
    fn default() -> Self {
        Purity::Pure
    }
}

impl Default for Execution {
    fn default() -> Self {
        Execution::Cpu
    }
}

impl Default for Allocation {
    fn default() -> Self {
        Allocation::None
    }
}

impl Default for Concurrency {
    fn default() -> Self {
        Concurrency::Concurrent
    }
}

impl Type {
    pub fn integer_kind(&self) -> Option<IntegerKind> {
        match self {
            Type::Int8 => Some(IntegerKind::I8),
            Type::Int16 => Some(IntegerKind::I16),
            Type::Int32 => Some(IntegerKind::I32),
            Type::Int64 => Some(IntegerKind::I64),
            Type::UInt8 => Some(IntegerKind::U8),
            Type::UInt16 => Some(IntegerKind::U16),
            Type::UInt32 => Some(IntegerKind::U32),
            Type::UInt64 => Some(IntegerKind::U64),
            _ => None,
        }
    }

    pub fn is_integer(&self) -> bool {
        self.integer_kind().is_some()
    }

    pub fn is_integer_like(&self) -> bool {
        self.is_integer() || matches!(self, Type::IntLiteral(_))
    }
}

impl IntegerKind {
    pub fn is_signed(&self) -> bool {
        matches!(self, IntegerKind::I8 | IntegerKind::I16 | IntegerKind::I32 | IntegerKind::I64)
    }

    pub fn min_value(&self) -> i128 {
        match self {
            IntegerKind::I8 => i8::MIN as i128,
            IntegerKind::I16 => i16::MIN as i128,
            IntegerKind::I32 => i32::MIN as i128,
            IntegerKind::I64 => i64::MIN as i128,
            IntegerKind::U8 | IntegerKind::U16 | IntegerKind::U32 | IntegerKind::U64 => 0,
        }
    }

    pub fn max_value(&self) -> i128 {
        match self {
            IntegerKind::I8 => i8::MAX as i128,
            IntegerKind::I16 => i16::MAX as i128,
            IntegerKind::I32 => i32::MAX as i128,
            IntegerKind::I64 => i64::MAX as i128,
            IntegerKind::U8 => u8::MAX as i128,
            IntegerKind::U16 => u16::MAX as i128,
            IntegerKind::U32 => u32::MAX as i128,
            IntegerKind::U64 => u64::MAX as i128,
        }
    }

    pub fn to_type(&self) -> Type {
        match self {
            IntegerKind::I8 => Type::Int8,
            IntegerKind::I16 => Type::Int16,
            IntegerKind::I32 => Type::Int32,
            IntegerKind::I64 => Type::Int64,
            IntegerKind::U8 => Type::UInt8,
            IntegerKind::U16 => Type::UInt16,
            IntegerKind::U32 => Type::UInt32,
            IntegerKind::U64 => Type::UInt64,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    // Literals
    IntLit(i128, Span),
    FloatLit(f64, Span),
    BoolLit(bool, Span),
    StringLit(String, Span),
    CharLit(char, Span),
    Unit(Span),

    // Variables
    Var(String, Span),

    // Function application
    App(Box<Expr>, Box<Expr>, Span),

    // Lambda (for higher-order functions)
    Lambda(Vec<String>, Box<Expr>, Span),

    // Let binding
    Let(String, Box<Expr>, Box<Expr>, Span),

    // If expression
    If(Box<Expr>, Box<Expr>, Box<Expr>, Span),

    // Match expression (pattern matching)
    Match(Box<Expr>, Vec<MatchArm>, Span),

    // Binary operations
    BinOp(BinOp, Box<Expr>, Box<Expr>, Span),

    // Unary operations
    UnOp(UnOp, Box<Expr>, Span),

    // Unsafe block
    Unsafe(Box<Expr>, Span),

    // Array literal
    ArrayLit(Vec<Expr>, Span),

    // Array indexing
    ArrayIndex(Box<Expr>, Box<Expr>, Span),

    // Constructor application
    Constructor(String, Vec<Expr>, Span),

    // Parallel primitives
    ParFor(Box<Expr>, Box<Expr>, Box<Expr>, Span), // (start, end, body)
    ParMap(Box<Expr>, Box<Expr>, Span), // (fn, array)
    ParMapInplace(Box<Expr>, Box<Expr>, Box<Expr>, Span), // (fn, src, dst)

    // Async/Task primitives
    Async(Box<Expr>, Span),
    Await(Box<Expr>, Span),

    // Actor primitives
    Spawn(Box<Expr>, Span), // Spawn actor
    Send(Box<Expr>, Box<Expr>, Span), // Send message to actor
    Receive(Span), // Receive message in actor

    // Memory primitives
    NewArena(Span),
    AllocArray(Box<Expr>, Box<Expr>, Space, Span), // (arena, size, space)

    // GPU primitives
    GpuKernel(Box<Expr>, Span),
    CpuToGpu(Box<Expr>, Span),
    GpuToCpu(Box<Expr>, Span),

    // Debug primitives
    Log(Box<Expr>, Span),
    Assert(Box<Expr>, Box<Expr>, Span), // (condition, message)

    // Sequence (for imperative-style code in IO contexts)
    Seq(Vec<Expr>, Span),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add, Sub, Mul, Div, Mod,
    Eq, Ne, Lt, Le, Gt, Ge,
    And, Or,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Not,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub body: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    Wildcard,
    Var(String),
    IntLit(i128),
    BoolLit(bool),
    Constructor(String, Vec<Pattern>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Span { start, end }
    }

    pub fn merge(&self, other: &Span) -> Self {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::IntLit(_, s) | Expr::FloatLit(_, s) | Expr::BoolLit(_, s) |
            Expr::StringLit(_, s) | Expr::CharLit(_, s) | Expr::Unit(s) |
            Expr::Var(_, s) | Expr::Unsafe(_, s) | Expr::ArrayLit(_, s) |
            Expr::NewArena(s) | Expr::Receive(s) | Expr::Log(_, s) |
            Expr::Async(_, s) | Expr::Await(_, s) | Expr::Spawn(_, s) |
            Expr::GpuKernel(_, s) | Expr::CpuToGpu(_, s) | Expr::GpuToCpu(_, s) |
            Expr::Seq(_, s) => *s,

            Expr::App(_, _, s) | Expr::Lambda(_, _, s) | Expr::Let(_, _, _, s) |
            Expr::If(_, _, _, s) | Expr::Match(_, _, s) | Expr::BinOp(_, _, _, s) |
            Expr::UnOp(_, _, s) | Expr::ArrayIndex(_, _, s) | Expr::Constructor(_, _, s) |
            Expr::ParFor(_, _, _, s) | Expr::ParMap(_, _, s) | Expr::ParMapInplace(_, _, _, s) |
            Expr::Send(_, _, s) | Expr::AllocArray(_, _, _, s) | Expr::Assert(_, _, s) => *s,
        }
    }
}

impl EffectSet {
    pub fn new() -> Self {
        EffectSet::default()
    }

    pub fn pure_cpu_none() -> Self {
        EffectSet {
            purity: Purity::Pure,
            execution: Execution::Cpu,
            allocation: Allocation::None,
            concurrency: Concurrency::Concurrent,
        }
    }

    pub fn with_purity(mut self, purity: Purity) -> Self {
        self.purity = purity;
        self
    }

    pub fn with_execution(mut self, execution: Execution) -> Self {
        self.execution = execution;
        self
    }

    pub fn with_allocation(mut self, allocation: Allocation) -> Self {
        self.allocation = allocation;
        self
    }

    pub fn with_concurrency(mut self, concurrency: Concurrency) -> Self {
        self.concurrency = concurrency;
        self
    }
}
