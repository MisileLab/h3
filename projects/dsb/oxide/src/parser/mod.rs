use crate::ast::{Expr, Literal, Op};
use crate::lexer::Token;

/// A very small hand-written parser sufficient to demonstrate
/// left-associative function application like `map f (filter p xs)`.
pub struct Parser {
    tokens: Vec<Token>,
    position: usize,
}

impl Parser {
    /// Create a new parser from a sequence of tokens.
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, position: 0 }
    }

    /// Parse a single expression.
    pub fn parse(&mut self) -> Result<Expr, String> {
        let expr = self.parse_expr()?;
        Ok(expr)
    }

    /// expression with binary operators and function application
    fn parse_expr(&mut self) -> Result<Expr, String> {
        self.parse_binary_expr(0)
    }

    /// Precedence climbing for binary operators.
    fn parse_binary_expr(&mut self, min_prec: u8) -> Result<Expr, String> {
        let mut left = self.parse_app_expr()?;

        loop {
            let (prec, op) = match self.peek().and_then(|t| Self::binary_precedence(t)) {
                Some((p, o)) if p >= min_prec => (p, o),
                _ => break,
            };

            // consume operator
            self.next();

            // parse the immediate right-hand side
            let mut right = self.parse_app_expr()?;

            // absorb any higher-precedence chains to the right
            loop {
                match self.peek().and_then(|t| Self::binary_precedence(t)) {
                    Some((next_prec, next_op)) if next_prec > prec => {
                        // consume higher-precedence op and its rhs
                        self.next();
                        let rhs2 = self.parse_app_expr()?;
                        right = Expr::BinaryOp(Box::new(right), next_op, Box::new(rhs2));
                    }
                    _ => break,
                }
            }

            left = Expr::BinaryOp(Box::new(left), op, Box::new(right));
        }

        Ok(left)
    }

    /// application := atom { atom }*
    /// Function application is left-associative: `f x y` -> App(App(f, [x]), [y])
    fn parse_app_expr(&mut self) -> Result<Expr, String> {
        let mut current = self.parse_atom()?;

        while let Some(tok) = self.peek() {
            if Self::is_atom_start(tok) {
                let arg = self.parse_atom()?;
                current = Expr::App {
                    func: Box::new(current),
                    args: vec![arg],
                };
                continue;
            }
            break;
        }

        Ok(current)
    }

    /// atom := identifier
    ///       | number
    ///       | string
    ///       | '(' expression ')'
    ///       | '|' ident+ '|' expression
    ///       | '#' atom
    fn parse_atom(&mut self) -> Result<Expr, String> {
        let token = self.next().cloned().ok_or_else(|| "Unexpected end of input".to_string())?;
        match token {
            Token::Identifier(name) => Ok(Expr::Variable(name)),
            Token::Nat(n) => Ok(Expr::Literal(Literal::Nat(n))),
            Token::Float(f) => Ok(Expr::Literal(Literal::Float(f))),
            Token::String(s) => Ok(Expr::Literal(Literal::String(s))),
            Token::LParen => {
                let expr = self.parse_expr()?;
                match self.next().cloned() {
                    Some(Token::RParen) => Ok(expr),
                    other => Err(format!("Expected ')' but found {:?}", other)),
                }
            }
            Token::Hash => {
                let inner = self.parse_atom()?;
                Ok(Expr::Move(Box::new(inner)))
            }
            Token::Pipe => {
                // parse one or more identifiers until the closing '|'
                let mut params: Vec<String> = Vec::new();
                loop {
                    match self.peek() {
                        Some(Token::Identifier(_name)) => {
                            // consume identifier
                            if let Some(Token::Identifier(n)) = self.next().cloned() {
                                params.push(n);
                            }
                        }
                        Some(Token::Pipe) => {
                            // consume closing pipe
                            self.next();
                            break;
                        }
                        other => {
                            return Err(format!(
                                "Expected parameter name or '|' to close lambda, found {:?}",
                                other
                            ));
                        }
                    }
                }

                if params.is_empty() {
                    return Err("Lambda must have at least one parameter".to_string());
                }

                let body = self.parse_expr()?;
                Ok(Expr::Lambda {
                    params,
                    body: Box::new(body),
                })
            }
            unexpected => Err(format!("Unexpected token in atom: {:?}", unexpected)),
        }
    }

    fn is_atom_start(token: &Token) -> bool {
        matches!(
            token,
            Token::Identifier(_)
                | Token::Nat(_)
                | Token::Float(_)
                | Token::String(_)
                | Token::LParen
                | Token::Hash
                | Token::Pipe
        )
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.position)
    }

    fn next(&mut self) -> Option<&Token> {
        let tok = self.tokens.get(self.position);
        if tok.is_some() {
            self.position += 1;
        }
        tok
    }

    fn binary_precedence(token: &Token) -> Option<(u8, Op)> {
        match token {
            Token::Star => Some((20, Op::Mul)),
            Token::Slash => Some((20, Op::Div)),
            Token::Plus => Some((10, Op::Add)),
            Token::Minus => Some((10, Op::Sub)),
            Token::GreaterThan => Some((5, Op::Gt)),
            Token::LessThan => Some((5, Op::Lt)),
            _ => None,
        }
    }
}
