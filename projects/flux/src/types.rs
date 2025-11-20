use crate::ast::*;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct TypeEnv {
    pub bindings: HashMap<String, TypeScheme>,
    pub parent: Option<Box<TypeEnv>>,
}

#[derive(Debug, Clone)]
pub struct TypeScheme {
    pub vars: Vec<String>,
    pub ty: Type,
    pub effects: EffectSet,
}

impl TypeEnv {
    pub fn new() -> Self {
        let mut env = TypeEnv {
            bindings: HashMap::new(),
            parent: None,
        };

        // Add built-in primitives with their types and effects
        env.add_primitive(
            "par_for",
            Type::Function(
                vec![
                    Type::Int32,
                    Type::Int32,
                    Type::Function(vec![Type::Int32], Box::new(Type::Unit), EffectSet::pure_cpu_none()),
                ],
                Box::new(Type::Unit),
                EffectSet::pure_cpu_none().with_concurrency(Concurrency::Concurrent),
            ),
            EffectSet::pure_cpu_none().with_concurrency(Concurrency::Concurrent),
        );

        env.add_primitive(
            "par_map",
            Type::Function(
                vec![
                    Type::Function(vec![Type::Var("a".into())], Box::new(Type::Var("b".into())), EffectSet::pure_cpu_none()),
                    Type::Array(Box::new(Type::Var("a".into())), Space::Cpu, ArenaRef::Anonymous),
                ],
                Box::new(Type::Array(Box::new(Type::Var("b".into())), Space::Cpu, ArenaRef::Anonymous)),
                EffectSet::pure_cpu_none()
                    .with_allocation(Allocation::Heap)
                    .with_concurrency(Concurrency::Concurrent),
            ),
            EffectSet::pure_cpu_none()
                .with_allocation(Allocation::Heap)
                .with_concurrency(Concurrency::Concurrent),
        );

        env.add_primitive(
            "par_map_inplace",
            Type::Function(
                vec![
                    Type::Function(vec![Type::Var("a".into())], Box::new(Type::Var("b".into())), EffectSet::pure_cpu_none()),
                    Type::Array(Box::new(Type::Var("a".into())), Space::Cpu, ArenaRef::Anonymous),
                    Type::Array(Box::new(Type::Var("b".into())), Space::Cpu, ArenaRef::Anonymous),
                ],
                Box::new(Type::Unit),
                EffectSet::pure_cpu_none().with_concurrency(Concurrency::Concurrent),
            ),
            EffectSet::pure_cpu_none().with_concurrency(Concurrency::Concurrent),
        );

        env.add_primitive(
            "async",
            Type::Function(
                vec![Type::Function(vec![Type::Unit], Box::new(Type::Var("a".into())), EffectSet::new().with_purity(Purity::Io))],
                Box::new(Type::Task(Box::new(Type::Var("a".into())))),
                EffectSet::new()
                    .with_purity(Purity::Io)
                    .with_concurrency(Concurrency::Concurrent),
            ),
            EffectSet::new()
                .with_purity(Purity::Io)
                .with_concurrency(Concurrency::Concurrent),
        );

        env.add_primitive(
            "await",
            Type::Function(
                vec![Type::Task(Box::new(Type::Var("a".into())))],
                Box::new(Type::Var("a".into())),
                EffectSet::new().with_purity(Purity::Io),
            ),
            EffectSet::new().with_purity(Purity::Io),
        );

        env.add_primitive(
            "spawn",
            Type::Function(
                vec![Type::Var("a".into())],
                Box::new(Type::Actor(Box::new(Type::Var("a".into())))),
                EffectSet::new()
                    .with_purity(Purity::Io)
                    .with_allocation(Allocation::Heap)
                    .with_concurrency(Concurrency::Concurrent),
            ),
            EffectSet::new()
                .with_purity(Purity::Io)
                .with_allocation(Allocation::Heap)
                .with_concurrency(Concurrency::Concurrent),
        );

        env.add_primitive(
            "send",
            Type::Function(
                vec![Type::Actor(Box::new(Type::Var("a".into()))), Type::Var("a".into())],
                Box::new(Type::Unit),
                EffectSet::new()
                    .with_purity(Purity::Io)
                    .with_concurrency(Concurrency::Concurrent),
            ),
            EffectSet::new()
                .with_purity(Purity::Io)
                .with_concurrency(Concurrency::Concurrent),
        );

        env.add_primitive(
            "receive",
            Type::Function(
                vec![],
                Box::new(Type::Var("a".into())),
                EffectSet::new()
                    .with_purity(Purity::Io)
                    .with_concurrency(Concurrency::Concurrent),
            ),
            EffectSet::new()
                .with_purity(Purity::Io)
                .with_concurrency(Concurrency::Concurrent),
        );

        env.add_primitive(
            "new_arena",
            Type::Function(
                vec![],
                Box::new(Type::Arena),
                EffectSet::pure_cpu_none().with_allocation(Allocation::Arena),
            ),
            EffectSet::pure_cpu_none().with_allocation(Allocation::Arena),
        );

        env.add_primitive(
            "alloc_array",
            Type::Function(
                vec![Type::Arena, Type::Int32],
                Box::new(Type::Array(
                    Box::new(Type::Var("a".into())),
                    Space::Cpu,
                    ArenaRef::Anonymous,
                )),
                EffectSet::pure_cpu_none().with_allocation(Allocation::Arena),
            ),
            EffectSet::pure_cpu_none().with_allocation(Allocation::Arena),
        );

        env.add_primitive(
            "log",
            Type::Function(
                vec![Type::Var("a".into())],
                Box::new(Type::Unit),
                EffectSet::new().with_purity(Purity::Debug),
            ),
            EffectSet::new().with_purity(Purity::Debug),
        );

        env.add_primitive(
            "assert",
            Type::Function(
                vec![Type::Bool, Type::String],
                Box::new(Type::Unit),
                EffectSet::new().with_purity(Purity::Debug),
            ),
            EffectSet::new().with_purity(Purity::Debug),
        );

        env
    }

    fn add_primitive(&mut self, name: &str, ty: Type, effects: EffectSet) {
        self.bindings.insert(
            name.to_string(),
            TypeScheme {
                vars: vec![],
                ty,
                effects,
            },
        );
    }

    pub fn extend(&self) -> Self {
        TypeEnv {
            bindings: HashMap::new(),
            parent: Some(Box::new(self.clone())),
        }
    }

    pub fn bind(&mut self, name: String, scheme: TypeScheme) {
        self.bindings.insert(name, scheme);
    }

    pub fn lookup(&self, name: &str) -> Option<&TypeScheme> {
        self.bindings
            .get(name)
            .or_else(|| self.parent.as_ref().and_then(|p| p.lookup(name)))
    }
}

impl Default for TypeEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeScheme {
    pub fn mono(ty: Type, effects: EffectSet) -> Self {
        TypeScheme {
            vars: vec![],
            ty,
            effects,
        }
    }
}

#[derive(Debug, Clone)]
pub struct UnifyError {
    pub message: String,
    pub span: Option<Span>,
}

impl UnifyError {
    fn new(msg: impl Into<String>, span: Option<Span>) -> Self {
        UnifyError {
            message: msg.into(),
            span,
        }
    }
}

#[derive(Debug, Clone)]
struct IntegerLiteralRecord {
    value: i128,
    span: Span,
    bound_kind: Option<IntegerKind>,
}

// Type unification and inference utilities
pub struct TypeInferencer {
    next_var: usize,
    substitution: HashMap<String, Type>,
    next_literal_id: LiteralId,
    literal_records: HashMap<LiteralId, IntegerLiteralRecord>,
    literal_links: HashMap<LiteralId, Vec<LiteralId>>,
}

impl TypeInferencer {
    pub fn new() -> Self {
        TypeInferencer {
            next_var: 0,
            substitution: HashMap::new(),
            next_literal_id: 0,
            literal_records: HashMap::new(),
            literal_links: HashMap::new(),
        }
    }

    pub fn reset(&mut self) {
        self.next_var = 0;
        self.substitution.clear();
        self.next_literal_id = 0;
        self.literal_records.clear();
        self.literal_links.clear();
    }

    pub fn fresh_var(&mut self) -> Type {
        let var = format!("t{}", self.next_var);
        self.next_var += 1;
        Type::Var(var)
    }

    pub fn register_literal(&mut self, value: i128, span: Span) -> LiteralId {
        let id = self.next_literal_id;
        self.next_literal_id += 1;
        self.literal_records.insert(
            id,
            IntegerLiteralRecord {
                value,
                span,
                bound_kind: None,
            },
        );
        id
    }

    fn literal_span(&self, id: LiteralId) -> Option<Span> {
        self.literal_records.get(&id).map(|rec| rec.span)
    }

    fn link_literals(&mut self, a: LiteralId, b: LiteralId) {
        if a == b {
            return;
        }
        self.literal_links.entry(a).or_default().push(b);
        self.literal_links.entry(b).or_default().push(a);
    }

    fn bind_literal(&mut self, id: LiteralId, kind: IntegerKind) -> Result<(), UnifyError> {
        let mut stack = vec![id];
        let mut visited = HashSet::new();

        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }

            let record = self.literal_records.get_mut(&current).ok_or_else(|| {
                UnifyError::new(
                    format!("Unknown integer literal with id {}", current),
                    None,
                )
            })?;

            if let Some(existing) = record.bound_kind {
                if existing != kind {
                    return Err(UnifyError::new(
                        format!(
                            "Conflicting integer bindings: {:?} vs {:?}",
                            existing, kind
                        ),
                        Some(record.span),
                    ));
                }
            } else {
                if record.value < kind.min_value() || record.value > kind.max_value() {
                    return Err(UnifyError::new(
                        format!(
                            "Integer literal {} out of range for {:?}",
                            record.value, kind
                        ),
                        Some(record.span),
                    ));
                }
                record.bound_kind = Some(kind);
            }

            if let Some(neighbors) = self.literal_links.get(&current) {
                stack.extend(neighbors.iter().copied());
            }
        }

        Ok(())
    }

    pub fn ensure_all_literals_bound(&mut self) -> Result<(), UnifyError> {
        let unbound: Vec<LiteralId> = self
            .literal_records
            .iter()
            .filter(|(_, record)| record.bound_kind.is_none())
            .map(|(id, _)| *id)
            .collect();

        for id in unbound {
            self.bind_literal(id, IntegerKind::I32)?;
        }

        Ok(())
    }

    pub fn resolved_literals(&self) -> Vec<(Span, IntegerKind)> {
        self.literal_records
            .values()
            .filter_map(|record| record.bound_kind.map(|kind| (record.span, kind)))
            .collect()
    }

    pub fn unify(&mut self, t1: &Type, t2: &Type) -> Result<(), UnifyError> {
        match (t1, t2) {
            (Type::IntLiteral(id1), Type::IntLiteral(id2)) => {
                self.link_literals(*id1, *id2);
                Ok(())
            }
            (Type::IntLiteral(id), other) | (other, Type::IntLiteral(id)) => {
                if let Some(kind) = other.integer_kind() {
                    self.bind_literal(*id, kind)
                } else {
                    Err(UnifyError::new(
                        format!("Expected integer type but found {:?}", other),
                        self.literal_span(*id),
                    ))
                }
            }
            _ if t1.integer_kind().is_some() && t1.integer_kind() == t2.integer_kind() => Ok(()),
            (Type::Float32, Type::Float32)
            | (Type::Float64, Type::Float64)
            | (Type::Bool, Type::Bool)
            | (Type::Unit, Type::Unit)
            | (Type::String, Type::String)
            | (Type::Char, Type::Char)
            | (Type::Arena, Type::Arena) => Ok(()),

            (Type::Var(v), t) | (t, Type::Var(v)) => {
                if let Some(bound) = self.substitution.get(v).cloned() {
                    self.unify(&bound, t)
                } else {
                    self.substitution.insert(v.clone(), t.clone());
                    Ok(())
                }
            }

            (Type::Function(p1, r1, e1), Type::Function(p2, r2, e2)) => {
                if p1.len() != p2.len() {
                    return Err(UnifyError::new(
                        format!("Function arity mismatch: {} vs {}", p1.len(), p2.len()),
                        None,
                    ));
                }

                for (pt1, pt2) in p1.iter().zip(p2.iter()) {
                    self.unify(pt1, pt2)?;
                }

                self.unify(r1, r2)?;

                if e1 != e2 {
                    return Err(UnifyError::new(
                        format!("Effect mismatch: {:?} vs {:?}", e1, e2),
                        None,
                    ));
                }

                Ok(())
            }

            (Type::Array(e1, s1, _), Type::Array(e2, s2, _)) => {
                self.unify(e1, e2)?;
                if s1 != s2 {
                    return Err(UnifyError::new(
                        format!("Space mismatch: {:?} vs {:?}", s1, s2),
                        None,
                    ));
                }
                // Arena references can be flexible
                Ok(())
            }

            (Type::Task(t1), Type::Task(t2)) | (Type::Actor(t1), Type::Actor(t2)) => {
                self.unify(t1, t2)
            }

            (Type::App(n1, args1), Type::App(n2, args2)) => {
                if n1 != n2 {
                    return Err(UnifyError::new(
                        format!("Type constructor mismatch: {} vs {}", n1, n2),
                        None,
                    ));
                }
                if args1.len() != args2.len() {
                    return Err(UnifyError::new("Type argument count mismatch", None));
                }
                for (a1, a2) in args1.iter().zip(args2.iter()) {
                    self.unify(a1, a2)?;
                }
                Ok(())
            }

            _ => Err(UnifyError::new(
                format!("Type mismatch: {:?} vs {:?}", t1, t2),
                None,
            )),
        }
    }

    pub fn apply_substitution(&self, ty: &Type) -> Type {
        match ty {
            Type::Var(v) => self
                .substitution
                .get(v)
                .map(|t| self.apply_substitution(t))
                .unwrap_or_else(|| ty.clone()),
            Type::Function(params, ret, effects) => Type::Function(
                params.iter().map(|p| self.apply_substitution(p)).collect(),
                Box::new(self.apply_substitution(ret)),
                effects.clone(),
            ),
            Type::Array(elem, space, arena) => Type::Array(
                Box::new(self.apply_substitution(elem)),
                space.clone(),
                arena.clone(),
            ),
            Type::Task(inner) => Type::Task(Box::new(self.apply_substitution(inner))),
            Type::Actor(inner) => Type::Actor(Box::new(self.apply_substitution(inner))),
            Type::App(name, args) => Type::App(
                name.clone(),
                args.iter().map(|a| self.apply_substitution(a)).collect(),
            ),
            Type::IntLiteral(id) => self
                .literal_records
                .get(id)
                .and_then(|record| record.bound_kind.map(|kind| kind.to_type()))
                .unwrap_or_else(|| ty.clone()),
            _ => ty.clone(),
        }
    }
}

impl Default for TypeInferencer {
    fn default() -> Self {
        Self::new()
    }
}
