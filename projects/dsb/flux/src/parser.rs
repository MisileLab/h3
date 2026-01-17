use crate::ast::*;
use crate::lexer::{Lexer, Token};

pub struct Parser {
    tokens: Vec<Token>,
    position: usize,
}

#[derive(Debug)]
pub struct ParseError {
    pub message: String,
}

impl ParseError {
    fn new(msg: impl Into<String>) -> Self {
        ParseError {
            message: msg.into(),
        }
    }
}

type ParseResult<T> = Result<T, ParseError>;

impl Parser {
    pub fn new(input: &str) -> Self {
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        Parser { tokens, position: 0 }
    }

    fn current(&self) -> &Token {
        self.tokens.get(self.position).unwrap_or(&Token::Eof)
    }

    fn advance(&mut self) {
        if self.position < self.tokens.len() {
            self.position += 1;
        }
    }

    fn expect(&mut self, expected: Token) -> ParseResult<()> {
        if self.current() == &expected {
            self.advance();
            Ok(())
        } else {
            Err(ParseError::new(format!(
                "Expected {:?}, found {:?}",
                expected,
                self.current()
            )))
        }
    }

    pub fn parse_module(&mut self) -> ParseResult<Module> {
        let mut declarations = Vec::new();

        while self.current() != &Token::Eof {
            // Skip newlines between declarations
            while self.current() == &Token::Newline {
                self.advance();
            }

            // Check for EOF again after skipping newlines
            if self.current() == &Token::Eof {
                break;
            }

            declarations.push(self.parse_declaration()?);
        }

        Ok(Module { declarations })
    }

    fn parse_declaration(&mut self) -> ParseResult<Declaration> {
        match self.current() {
            Token::Data => Ok(Declaration::DataType(self.parse_data_type()?)),
            Token::Type => Ok(Declaration::TypeAlias(self.parse_type_alias()?)),
            Token::Ident(_) => Ok(Declaration::Function(self.parse_function()?)),
            _ => Err(ParseError::new(format!(
                "Unexpected token in declaration: {:?}",
                self.current()
            ))),
        }
    }

    fn parse_data_type(&mut self) -> ParseResult<DataTypeDecl> {
        self.expect(Token::Data)?;

        let name = match self.current() {
            Token::Ident(s) => {
                let name = s.clone();
                self.advance();
                name
            }
            _ => return Err(ParseError::new("Expected type name")),
        };

        let mut type_params = Vec::new();
        while let Token::Ident(s) = self.current() {
            if s.chars().next().unwrap().is_lowercase() {
                type_params.push(s.clone());
                self.advance();
            } else {
                break;
            }
        }

        self.expect(Token::Eq)?;

        let mut constructors = Vec::new();

        loop {
            let ctor_name = match self.current() {
                Token::Ident(s) => {
                    let name = s.clone();
                    self.advance();
                    name
                }
                _ => break,
            };

            let mut fields = Vec::new();
            while !matches!(self.current(), Token::Pipe | Token::Eof | Token::Ident(_)) {
                fields.push(self.parse_type()?);
            }

            constructors.push(Constructor {
                name: ctor_name,
                fields,
            });

            if self.current() == &Token::Pipe {
                self.advance();
            } else {
                break;
            }
        }

        Ok(DataTypeDecl {
            name,
            type_params,
            constructors,
        })
    }

    fn parse_type_alias(&mut self) -> ParseResult<TypeAliasDecl> {
        self.expect(Token::Type)?;

        let name = match self.current() {
            Token::Ident(s) => {
                let name = s.clone();
                self.advance();
                name
            }
            _ => return Err(ParseError::new("Expected type name")),
        };

        let mut type_params = Vec::new();
        while let Token::Ident(s) = self.current() {
            if s.chars().next().unwrap().is_lowercase() {
                type_params.push(s.clone());
                self.advance();
            } else {
                break;
            }
        }

        self.expect(Token::Eq)?;

        let aliased_type = self.parse_type()?;

        Ok(TypeAliasDecl {
            name,
            type_params,
            aliased_type,
        })
    }

    fn parse_function(&mut self) -> ParseResult<FunctionDecl> {
        let start = self.position;

        let name = match self.current() {
            Token::Ident(s) => {
                let name = s.clone();
                self.advance();
                name
            }
            _ => return Err(ParseError::new("Expected function name")),
        };

        let type_sig = if self.current() == &Token::Colon {
            self.advance();
            Some(self.parse_type_signature()?)
        } else {
            None
        };

        // Skip any newlines between type signature and function body
        while self.current() == &Token::Newline {
            self.advance();
        }

        // If we just parsed a type signature, we expect the function name again
        // for the implementation line (e.g., "foo x = ...")
        if type_sig.is_some() {
            if let Token::Ident(s) = self.current() {
                if s == &name {
                    self.advance(); // Skip the repeated function name
                } else {
                    return Err(ParseError::new(&format!(
                        "Expected function name '{}' in implementation, found '{}'",
                        name, s
                    )));
                }
            } else {
                return Err(ParseError::new(&format!(
                    "Expected function name '{}' in implementation",
                    name
                )));
            }
        }

        // Parse function parameters and body
        let mut params = Vec::new();

        while let Token::Ident(s) = self.current() {
            params.push(s.clone());
            self.advance();
        }

        self.expect(Token::Eq)?;

        let body = self.parse_expr()?;

        Ok(FunctionDecl {
            name,
            type_sig,
            params,
            body,
            span: Span::new(start, self.position),
        })
    }

    fn parse_type_signature(&mut self) -> ParseResult<TypeSignature> {
        let mut param_types = Vec::new();

        loop {
            let ty = self.parse_type()?;

            if self.current() == &Token::Arrow {
                param_types.push(ty);
                self.advance();
            } else {
                // Last type is return type
                let return_type = Box::new(ty);

                let effects = if self.current() == &Token::Bang {
                    self.advance();
                    Some(self.parse_effect_set()?)
                } else {
                    None
                };

                return Ok(TypeSignature {
                    param_types,
                    return_type,
                    effects,
                });
            }
        }
    }

    fn parse_type(&mut self) -> ParseResult<Type> {
        match self.current() {
            Token::Ident(s) if s == "i8" => {
                self.advance();
                Ok(Type::Int8)
            }
            Token::Ident(s) if s == "i16" => {
                self.advance();
                Ok(Type::Int16)
            }
            Token::Ident(s) if s == "i32" => {
                self.advance();
                Ok(Type::Int32)
            }
            Token::Ident(s) if s == "i64" => {
                self.advance();
                Ok(Type::Int64)
            }
            Token::Ident(s) if s == "u8" => {
                self.advance();
                Ok(Type::UInt8)
            }
            Token::Ident(s) if s == "u16" => {
                self.advance();
                Ok(Type::UInt16)
            }
            Token::Ident(s) if s == "u32" => {
                self.advance();
                Ok(Type::UInt32)
            }
            Token::Ident(s) if s == "u64" => {
                self.advance();
                Ok(Type::UInt64)
            }
            Token::Ident(s) if s == "f32" => {
                self.advance();
                Ok(Type::Float32)
            }
            Token::Ident(s) if s == "f64" => {
                self.advance();
                Ok(Type::Float64)
            }
            Token::Ident(s) if s == "Bool" => {
                self.advance();
                Ok(Type::Bool)
            }
            Token::Ident(s) if s == "Unit" => {
                self.advance();
                Ok(Type::Unit)
            }
            Token::Ident(s) if s == "String" => {
                self.advance();
                Ok(Type::String)
            }
            Token::Ident(s) if s == "Char" => {
                self.advance();
                Ok(Type::Char)
            }
            Token::Ident(s) if s == "Arena" => {
                self.advance();
                Ok(Type::Arena)
            }
            Token::Ident(s) if s == "Array" => {
                self.advance();
                // Array<T, Space, Arena>
                let elem_type = Box::new(self.parse_type()?);
                let space = self.parse_space()?;
                let arena = self.parse_arena_ref()?;
                Ok(Type::Array(elem_type, space, arena))
            }
            Token::Ident(s) if s == "Task" => {
                self.advance();
                let inner = Box::new(self.parse_type()?);
                Ok(Type::Task(inner))
            }
            Token::Ident(s) if s == "Actor" => {
                self.advance();
                let inner = Box::new(self.parse_type()?);
                Ok(Type::Actor(inner))
            }
            Token::Ident(s) if s.chars().next().unwrap().is_lowercase() => {
                let var = s.clone();
                self.advance();
                Ok(Type::Var(var))
            }
            Token::Ident(s) => {
                let name = s.clone();
                self.advance();

                let mut args = Vec::new();
                while !matches!(
                    self.current(),
                    Token::Arrow | Token::Bang | Token::RParen | Token::Comma | Token::Eof
                ) {
                    args.push(self.parse_type()?);
                }

                if args.is_empty() {
                    Ok(Type::App(name, vec![]))
                } else {
                    Ok(Type::App(name, args))
                }
            }
            Token::LParen => {
                self.advance();
                let ty = self.parse_type()?;
                self.expect(Token::RParen)?;
                Ok(ty)
            }
            _ => Err(ParseError::new(format!(
                "Expected type, found {:?}",
                self.current()
            ))),
        }
    }

    fn parse_space(&mut self) -> ParseResult<Space> {
        match self.current() {
            Token::Cpu => {
                self.advance();
                Ok(Space::Cpu)
            }
            Token::Gpu => {
                self.advance();
                Ok(Space::Gpu)
            }
            _ => Ok(Space::Cpu), // Default to CPU
        }
    }

    fn parse_arena_ref(&mut self) -> ParseResult<ArenaRef> {
        match self.current() {
            Token::Ident(s) => {
                let name = s.clone();
                self.advance();
                Ok(ArenaRef::Named(name))
            }
            _ => Ok(ArenaRef::Anonymous),
        }
    }

    fn parse_effect_set(&mut self) -> ParseResult<EffectSet> {
        self.expect(Token::LBrace)?;

        let mut effect_set = EffectSet::default();

        loop {
            match self.current() {
                Token::Pure => {
                    effect_set.purity = Purity::Pure;
                    self.advance();
                }
                Token::Io => {
                    effect_set.purity = Purity::Io;
                    self.advance();
                }
                Token::State => {
                    effect_set.purity = Purity::State;
                    self.advance();
                }
                Token::Debug => {
                    effect_set.purity = Purity::Debug;
                    self.advance();
                }
                Token::Cpu => {
                    effect_set.execution = Execution::Cpu;
                    self.advance();
                }
                Token::Gpu => {
                    effect_set.execution = Execution::Gpu;
                    self.advance();
                }
                Token::Alloc => {
                    self.advance();
                    match self.current() {
                        Token::None_ => {
                            effect_set.allocation = Allocation::None;
                            self.advance();
                        }
                        Token::Arena => {
                            effect_set.allocation = Allocation::Arena;
                            self.advance();
                        }
                        Token::Heap => {
                            effect_set.allocation = Allocation::Heap;
                            self.advance();
                        }
                        _ => return Err(ParseError::new("Expected allocation kind")),
                    }
                }
                Token::Concurrent => {
                    effect_set.concurrency = Concurrency::Concurrent;
                    self.advance();
                }
                Token::Single => {
                    effect_set.concurrency = Concurrency::Single;
                    self.advance();
                }
                Token::Comma => {
                    self.advance();
                }
                Token::RBrace => {
                    self.advance();
                    break;
                }
                _ => {
                    return Err(ParseError::new(format!(
                        "Unexpected token in effect set: {:?}",
                        self.current()
                    )))
                }
            }
        }

        Ok(effect_set)
    }

    fn parse_expr(&mut self) -> ParseResult<Expr> {
        // Skip leading newlines in expressions
        while self.current() == &Token::Newline {
            self.advance();
        }
        self.parse_let_expr()
    }

    fn parse_let_expr(&mut self) -> ParseResult<Expr> {
        if self.current() == &Token::Let {
            let start = self.position;
            self.advance();

            let name = match self.current() {
                Token::Ident(s) => {
                    let n = s.clone();
                    self.advance();
                    n
                }
                _ => return Err(ParseError::new("Expected identifier in let binding")),
            };

            self.expect(Token::Eq)?;

            let value = self.parse_expr()?;

            self.expect(Token::In)?;

            let body = self.parse_expr()?;

            Ok(Expr::Let(
                name,
                Box::new(value),
                Box::new(body),
                Span::new(start, self.position),
            ))
        } else {
            self.parse_if_expr()
        }
    }

    fn parse_if_expr(&mut self) -> ParseResult<Expr> {
        if self.current() == &Token::If {
            let start = self.position;
            self.advance();

            let cond = self.parse_logical_or()?;

            // Skip newlines before 'then'
            while self.current() == &Token::Newline {
                self.advance();
            }

            self.expect(Token::Then)?;

            let then_branch = self.parse_expr()?;

            // Skip newlines before 'else'
            while self.current() == &Token::Newline {
                self.advance();
            }

            self.expect(Token::Else)?;

            let else_branch = self.parse_expr()?;

            Ok(Expr::If(
                Box::new(cond),
                Box::new(then_branch),
                Box::new(else_branch),
                Span::new(start, self.position),
            ))
        } else if self.current() == &Token::Match {
            self.parse_match_expr()
        } else if self.current() == &Token::Unsafe {
            self.parse_unsafe_expr()
        } else {
            self.parse_logical_or()
        }
    }

    fn parse_match_expr(&mut self) -> ParseResult<Expr> {
        let start = self.position;
        self.expect(Token::Match)?;

        let scrutinee = self.parse_logical_or()?;

        self.expect(Token::With)?;

        let mut arms = Vec::new();

        while self.current() == &Token::Pipe {
            self.advance();

            let pattern = self.parse_pattern()?;

            self.expect(Token::Arrow)?;

            let body = self.parse_expr()?;

            arms.push(MatchArm { pattern, body });
        }

        Ok(Expr::Match(
            Box::new(scrutinee),
            arms,
            Span::new(start, self.position),
        ))
    }

    fn parse_unsafe_expr(&mut self) -> ParseResult<Expr> {
        let start = self.position;
        self.expect(Token::Unsafe)?;
        self.expect(Token::LBrace)?;

        let expr = self.parse_expr()?;

        self.expect(Token::RBrace)?;

        Ok(Expr::Unsafe(Box::new(expr), Span::new(start, self.position)))
    }

    fn parse_pattern(&mut self) -> ParseResult<Pattern> {
        match self.current() {
            Token::Underscore => {
                self.advance();
                Ok(Pattern::Wildcard)
            }
            Token::IntLit(n) => {
                let val = *n;
                self.advance();
                Ok(Pattern::IntLit(val))
            }
            Token::BoolLit(b) => {
                let val = *b;
                self.advance();
                Ok(Pattern::BoolLit(val))
            }
            Token::Ident(s) if s.chars().next().unwrap().is_uppercase() => {
                let name = s.clone();
                self.advance();

                let mut patterns = Vec::new();
                while !matches!(self.current(), Token::Arrow | Token::Eof) {
                    patterns.push(self.parse_pattern()?);
                }

                Ok(Pattern::Constructor(name, patterns))
            }
            Token::Ident(s) => {
                let name = s.clone();
                self.advance();
                Ok(Pattern::Var(name))
            }
            _ => Err(ParseError::new("Expected pattern")),
        }
    }

    fn parse_logical_or(&mut self) -> ParseResult<Expr> {
        let mut left = self.parse_logical_and()?;

        while matches!(self.current(), Token::Or) {
            let start = left.span().start;
            self.advance();
            let right = self.parse_logical_and()?;
            left = Expr::BinOp(
                BinOp::Or,
                Box::new(left),
                Box::new(right),
                Span::new(start, self.position),
            );
        }

        Ok(left)
    }

    fn parse_logical_and(&mut self) -> ParseResult<Expr> {
        let mut left = self.parse_comparison()?;

        while matches!(self.current(), Token::And) {
            let start = left.span().start;
            self.advance();
            let right = self.parse_comparison()?;
            left = Expr::BinOp(
                BinOp::And,
                Box::new(left),
                Box::new(right),
                Span::new(start, self.position),
            );
        }

        Ok(left)
    }

    fn parse_comparison(&mut self) -> ParseResult<Expr> {
        let mut left = self.parse_additive()?;

        while matches!(
            self.current(),
            Token::Eq | Token::Ne | Token::Lt | Token::Le | Token::Gt | Token::Ge
        ) {
            let start = left.span().start;
            let op = match self.current() {
                Token::Eq => BinOp::Eq,
                Token::Ne => BinOp::Ne,
                Token::Lt => BinOp::Lt,
                Token::Le => BinOp::Le,
                Token::Gt => BinOp::Gt,
                Token::Ge => BinOp::Ge,
                _ => unreachable!(),
            };
            self.advance();
            let right = self.parse_additive()?;
            left = Expr::BinOp(
                op,
                Box::new(left),
                Box::new(right),
                Span::new(start, self.position),
            );
        }

        Ok(left)
    }

    fn parse_additive(&mut self) -> ParseResult<Expr> {
        let mut left = self.parse_multiplicative()?;

        while matches!(self.current(), Token::Plus | Token::Minus) {
            let start = left.span().start;
            let op = match self.current() {
                Token::Plus => BinOp::Add,
                Token::Minus => BinOp::Sub,
                _ => unreachable!(),
            };
            self.advance();
            let right = self.parse_multiplicative()?;
            left = Expr::BinOp(
                op,
                Box::new(left),
                Box::new(right),
                Span::new(start, self.position),
            );
        }

        Ok(left)
    }

    fn parse_multiplicative(&mut self) -> ParseResult<Expr> {
        let mut left = self.parse_unary()?;

        while matches!(self.current(), Token::Star | Token::Slash | Token::Percent) {
            let start = left.span().start;
            let op = match self.current() {
                Token::Star => BinOp::Mul,
                Token::Slash => BinOp::Div,
                Token::Percent => BinOp::Mod,
                _ => unreachable!(),
            };
            self.advance();
            let right = self.parse_unary()?;
            left = Expr::BinOp(
                op,
                Box::new(left),
                Box::new(right),
                Span::new(start, self.position),
            );
        }

        Ok(left)
    }

    fn parse_unary(&mut self) -> ParseResult<Expr> {
        match self.current() {
            Token::Minus => {
                let start = self.position;
                self.advance();
                let expr = self.parse_unary()?;
                match expr {
                    Expr::IntLit(value, _) => {
                        Ok(Expr::IntLit(-value, Span::new(start, self.position)))
                    }
                    _ => Ok(Expr::UnOp(
                        UnOp::Neg,
                        Box::new(expr),
                        Span::new(start, self.position),
                    )),
                }
            }
            Token::Not => {
                let start = self.position;
                self.advance();
                let expr = self.parse_unary()?;
                Ok(Expr::UnOp(
                    UnOp::Not,
                    Box::new(expr),
                    Span::new(start, self.position),
                ))
            }
            _ => self.parse_application(),
        }
    }

    fn parse_application(&mut self) -> ParseResult<Expr> {
        let mut left = self.parse_primary()?;

        loop {
            match self.current() {
                Token::LParen | Token::IntLit(_) | Token::BoolLit(_) | Token::StringLit(_)
                | Token::Ident(_) | Token::LBracket => {
                    let start = left.span().start;
                    let right = self.parse_primary()?;
                    left = Expr::App(
                        Box::new(left),
                        Box::new(right),
                        Span::new(start, self.position),
                    );
                }
                _ => break,
            }
        }

        Ok(left)
    }

    fn parse_primary(&mut self) -> ParseResult<Expr> {
        let start = self.position;

        match self.current().clone() {
            Token::IntLit(n) => {
                self.advance();
                Ok(Expr::IntLit(n, Span::new(start, self.position)))
            }
            Token::FloatLit(f) => {
                self.advance();
                Ok(Expr::FloatLit(f, Span::new(start, self.position)))
            }
            Token::BoolLit(b) => {
                self.advance();
                Ok(Expr::BoolLit(b, Span::new(start, self.position)))
            }
            Token::StringLit(s) => {
                self.advance();
                Ok(Expr::StringLit(s, Span::new(start, self.position)))
            }
            Token::CharLit(c) => {
                self.advance();
                Ok(Expr::CharLit(c, Span::new(start, self.position)))
            }
            Token::Ident(s) => {
                self.advance();
                Ok(Expr::Var(s, Span::new(start, self.position)))
            }
            Token::LParen => {
                self.advance();
                if self.current() == &Token::RParen {
                    self.advance();
                    return Ok(Expr::Unit(Span::new(start, self.position)));
                }
                let expr = self.parse_expr()?;
                self.expect(Token::RParen)?;
                Ok(expr)
            }
            Token::LBracket => {
                self.advance();
                let mut elements = Vec::new();

                while self.current() != &Token::RBracket {
                    elements.push(self.parse_expr()?);

                    if self.current() == &Token::Comma {
                        self.advance();
                    }
                }

                self.expect(Token::RBracket)?;
                Ok(Expr::ArrayLit(elements, Span::new(start, self.position)))
            }
            Token::ParFor => {
                self.advance();
                let range_start = self.parse_primary()?;
                let range_end = self.parse_primary()?;
                let body = self.parse_primary()?;
                Ok(Expr::ParFor(
                    Box::new(range_start),
                    Box::new(range_end),
                    Box::new(body),
                    Span::new(start, self.position),
                ))
            }
            Token::ParMap => {
                self.advance();
                let func = self.parse_primary()?;
                let array = self.parse_primary()?;
                Ok(Expr::ParMap(
                    Box::new(func),
                    Box::new(array),
                    Span::new(start, self.position),
                ))
            }
            Token::ParMapInplace => {
                self.advance();
                let func = self.parse_primary()?;
                let src = self.parse_primary()?;
                let dst = self.parse_primary()?;
                Ok(Expr::ParMapInplace(
                    Box::new(func),
                    Box::new(src),
                    Box::new(dst),
                    Span::new(start, self.position),
                ))
            }
            Token::Async => {
                self.advance();
                let expr = self.parse_primary()?;
                Ok(Expr::Async(Box::new(expr), Span::new(start, self.position)))
            }
            Token::Await => {
                self.advance();
                let expr = self.parse_primary()?;
                Ok(Expr::Await(Box::new(expr), Span::new(start, self.position)))
            }
            Token::Spawn => {
                self.advance();
                let expr = self.parse_primary()?;
                Ok(Expr::Spawn(Box::new(expr), Span::new(start, self.position)))
            }
            Token::Send => {
                self.advance();
                let actor = self.parse_primary()?;
                let message = self.parse_primary()?;
                Ok(Expr::Send(
                    Box::new(actor),
                    Box::new(message),
                    Span::new(start, self.position),
                ))
            }
            Token::Receive => {
                self.advance();
                Ok(Expr::Receive(Span::new(start, self.position)))
            }
            Token::NewArena => {
                self.advance();
                Ok(Expr::NewArena(Span::new(start, self.position)))
            }
            Token::AllocArray => {
                self.advance();
                let arena = self.parse_primary()?;
                let size = self.parse_primary()?;
                let space = match self.current() {
                    Token::Cpu => {
                        self.advance();
                        Space::Cpu
                    }
                    Token::Gpu => {
                        self.advance();
                        Space::Gpu
                    }
                    _ => Space::Cpu,
                };
                Ok(Expr::AllocArray(
                    Box::new(arena),
                    Box::new(size),
                    space,
                    Span::new(start, self.position),
                ))
            }
            Token::GpuKernel => {
                self.advance();
                let expr = self.parse_primary()?;
                Ok(Expr::GpuKernel(
                    Box::new(expr),
                    Span::new(start, self.position),
                ))
            }
            Token::CpuToGpu => {
                self.advance();
                let expr = self.parse_primary()?;
                Ok(Expr::CpuToGpu(
                    Box::new(expr),
                    Span::new(start, self.position),
                ))
            }
            Token::GpuToCpu => {
                self.advance();
                let expr = self.parse_primary()?;
                Ok(Expr::GpuToCpu(
                    Box::new(expr),
                    Span::new(start, self.position),
                ))
            }
            Token::Log => {
                self.advance();
                let expr = self.parse_primary()?;
                Ok(Expr::Log(Box::new(expr), Span::new(start, self.position)))
            }
            Token::Assert => {
                self.advance();
                let cond = self.parse_primary()?;
                let message = self.parse_primary()?;
                Ok(Expr::Assert(
                    Box::new(cond),
                    Box::new(message),
                    Span::new(start, self.position),
                ))
            }
            _ => Err(ParseError::new(format!(
                "Unexpected token: {:?}",
                self.current()
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_function() {
        let input = "add: i32 -> i32 -> i32\nadd x y = x + y";
        let mut parser = Parser::new(input);
        let module = parser.parse_module().unwrap();
        assert_eq!(module.declarations.len(), 1);
    }

    #[test]
    fn test_parse_effect_annotation() {
        let input = "foo: i32 -> i32 !{pure, cpu, alloc none}\nfoo x = x + 1";
        let mut parser = Parser::new(input);
        let module = parser.parse_module().unwrap();
        assert_eq!(module.declarations.len(), 1);
    }
}
