use crate::ast::*;
use crate::effect::EffectChecker;
use crate::types::*;
use std::collections::HashMap;

pub struct TypeChecker {
    effect_checker: EffectChecker,
    inferencer: TypeInferencer,
    literal_types: HashMap<Span, IntegerKind>,
}

#[derive(Debug)]
pub struct TypeCheckError {
    pub message: String,
    pub span: Span,
}

impl TypeCheckError {
    fn new(msg: impl Into<String>, span: Span) -> Self {
        TypeCheckError {
            message: msg.into(),
            span,
        }
    }
}

type TypeCheckResult<T> = Result<T, TypeCheckError>;

impl TypeChecker {
    pub fn new() -> Self {
        TypeChecker {
            effect_checker: EffectChecker::new(),
            inferencer: TypeInferencer::new(),
            literal_types: HashMap::new(),
        }
    }

    pub fn check_module(&mut self, module: &Module) -> TypeCheckResult<TypeEnv> {
        self.literal_types.clear();
        let mut env = TypeEnv::new();

        // First pass: collect all function signatures
        for decl in &module.declarations {
            match decl {
                Declaration::Function(func) => {
                    if let Some(sig) = &func.type_sig {
                        let effects = sig.effects.clone().unwrap_or_else(EffectSet::pure_cpu_none);

                        let func_type = if sig.param_types.is_empty() {
                            (*sig.return_type).clone()
                        } else {
                            Type::Function(
                                sig.param_types.clone(),
                                sig.return_type.clone(),
                                effects.clone(),
                            )
                        };

                        env.bind(
                            func.name.clone(),
                            TypeScheme::mono(func_type, effects),
                        );
                    }
                }
                Declaration::DataType(data) => {
                    // Register data type constructors
                    for ctor in &data.constructors {
                        let ctor_type = if ctor.fields.is_empty() {
                            Type::App(data.name.clone(), vec![])
                        } else {
                            Type::Function(
                                ctor.fields.clone(),
                                Box::new(Type::App(data.name.clone(), vec![])),
                                EffectSet::pure_cpu_none(),
                            )
                        };

                        env.bind(
                            ctor.name.clone(),
                            TypeScheme::mono(ctor_type, EffectSet::pure_cpu_none()),
                        );
                    }
                }
                Declaration::TypeAlias(_) => {
                    // Type aliases are handled during type resolution
                }
            }
        }

        // Second pass: check function bodies
        for decl in &module.declarations {
            if let Declaration::Function(func) = decl {
                self.check_function(func, &env)?;
            }
        }

        Ok(env)
    }

    fn check_function(&mut self, func: &FunctionDecl, env: &TypeEnv) -> TypeCheckResult<()> {
        self.inferencer.reset();
        let mut local_env = env.extend();

        // Get function effects
        let func_effects = func
            .type_sig
            .as_ref()
            .and_then(|sig| sig.effects.clone())
            .unwrap_or_else(EffectSet::pure_cpu_none);

        // Bind parameters
        if let Some(sig) = &func.type_sig {
            for (param, param_type) in func.params.iter().zip(&sig.param_types) {
                local_env.bind(
                    param.clone(),
                    TypeScheme::mono(param_type.clone(), EffectSet::pure_cpu_none()),
                );
            }
        }

        // Check body
        let (body_type, body_effects) = self.infer_expr(&func.body, &local_env, &func_effects)?;

        // Verify return type matches
        if let Some(sig) = &func.type_sig {
            self.unify_types(&body_type, &sig.return_type, func.span)?;
        }

        // Verify body effects are compatible with declared effects
        self.effect_checker
            .can_call(&func_effects, &body_effects)
            .map_err(|e| {
                TypeCheckError::new(
                    format!(
                        "Function '{}' body has incompatible effects: {}",
                        func.name, e
                    ),
                    func.span,
                )
            })?;

        self.inferencer
            .ensure_all_literals_bound()
            .map_err(|e| TypeCheckError::new(e.message, e.span.unwrap_or(func.span)))?;

        for (span, kind) in self.inferencer.resolved_literals() {
            self.literal_types.insert(span, kind);
        }

        Ok(())
    }

    fn infer_expr(
        &mut self,
        expr: &Expr,
        env: &TypeEnv,
        context_effects: &EffectSet,
    ) -> TypeCheckResult<(Type, EffectSet)> {
        match expr {
            Expr::IntLit(value, span) => {
                let literal_id = self.inferencer.register_literal(*value, *span);
                Ok((Type::IntLiteral(literal_id), EffectSet::pure_cpu_none()))
            }

            Expr::FloatLit(_, _) => Ok((Type::Float64, EffectSet::pure_cpu_none())),

            Expr::BoolLit(_, _) => Ok((Type::Bool, EffectSet::pure_cpu_none())),

            Expr::StringLit(_, _) => Ok((Type::String, EffectSet::pure_cpu_none())),

            Expr::CharLit(_, _) => Ok((Type::Char, EffectSet::pure_cpu_none())),

            Expr::Unit(_) => Ok((Type::Unit, EffectSet::pure_cpu_none())),

            Expr::Var(name, span) => {
                let scheme = env.lookup(name).ok_or_else(|| {
                    TypeCheckError::new(format!("Undefined variable: {}", name), *span)
                })?;

                Ok((scheme.ty.clone(), scheme.effects.clone()))
            }

            Expr::BinOp(op, left, right, span) => {
                let (left_type, left_eff) = self.infer_expr(left, env, context_effects)?;
                let (right_type, right_eff) = self.infer_expr(right, env, context_effects)?;

                let merged_effects = self.effect_checker.merge(&left_eff, &right_eff);

                // Type check based on operator
                let result_type = match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                        self.ensure_integer_like(&left_type, *span, "left operand")?;
                        self.ensure_integer_like(&right_type, *span, "right operand")?;

                        self.unify_types(&left_type, &right_type, *span)?;

                        self.inferencer.apply_substitution(&left_type)
                    }
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                        self.unify_types(&left_type, &right_type, *span)?;
                        Type::Bool
                    }
                    BinOp::And | BinOp::Or => {
                        self.unify_types(&left_type, &Type::Bool, *span)?;
                        self.unify_types(&right_type, &Type::Bool, *span)?;
                        Type::Bool
                    }
                };

                Ok((result_type, merged_effects))
            }

            Expr::UnOp(op, operand, span) => {
                let (operand_type, operand_eff) = self.infer_expr(operand, env, context_effects)?;

                let result_type = match op {
                    UnOp::Neg => {
                        self.ensure_integer_like(&operand_type, *span, "operand")?;
                        self.inferencer.apply_substitution(&operand_type)
                    }
                    UnOp::Not => {
                        self.unify_types(&operand_type, &Type::Bool, *span)?;
                        Type::Bool
                    }
                };

                Ok((result_type, operand_eff))
            }

            Expr::If(cond, then_br, else_br, span) => {
                let (cond_type, cond_eff) = self.infer_expr(cond, env, context_effects)?;
                self.unify_types(&cond_type, &Type::Bool, *span)?;

                let (then_type, then_eff) = self.infer_expr(then_br, env, context_effects)?;
                let (else_type, else_eff) = self.infer_expr(else_br, env, context_effects)?;

                self.unify_types(&then_type, &else_type, *span)?;

                let merged_effects = self
                    .effect_checker
                    .merge(&cond_eff, &self.effect_checker.merge(&then_eff, &else_eff));

                Ok((then_type, merged_effects))
            }

            Expr::Let(name, value, body, _) => {
                let (value_type, value_eff) = self.infer_expr(value, env, context_effects)?;

                let mut local_env = env.extend();
                local_env.bind(
                    name.clone(),
                    TypeScheme::mono(value_type, value_eff.clone()),
                );

                let (body_type, body_eff) = self.infer_expr(body, &local_env, context_effects)?;

                let merged_effects = self.effect_checker.merge(&value_eff, &body_eff);

                Ok((body_type, merged_effects))
            }

            Expr::App(func, arg, span) => {
                let (func_type, func_eff) = self.infer_expr(func, env, context_effects)?;
                let (arg_type, arg_eff) = self.infer_expr(arg, env, context_effects)?;

                match func_type {
                    Type::Function(param_types, return_type, call_effects) => {
                        if param_types.is_empty() {
                            return Err(TypeCheckError::new(
                                "Function has no parameters",
                                *span,
                            ));
                        }

                        // Unify first parameter with argument
                        self.unify_types(&param_types[0], &arg_type, *span)?;

                        // Check if we can call this function with current effects
                        self.effect_checker
                            .can_call(context_effects, &call_effects)
                            .map_err(|e| TypeCheckError::new(e, *span))?;

                        // Partial application
                        let result_type = if param_types.len() > 1 {
                            Type::Function(
                                param_types[1..].to_vec(),
                                return_type,
                                call_effects.clone(),
                            )
                        } else {
                            (*return_type).clone()
                        };

                        let merged_effects = self
                            .effect_checker
                            .merge(&func_eff, &self.effect_checker.merge(&arg_eff, &call_effects));

                        Ok((result_type, merged_effects))
                    }
                    _ => Err(TypeCheckError::new(
                        format!("Expected function type, got {:?}", func_type),
                        *span,
                    )),
                }
            }

            Expr::Lambda(params, body, _) => {
                let mut local_env = env.extend();

                // Create fresh type variables for parameters
                let mut param_types = Vec::new();
                for param in params {
                    let param_type = self.inferencer.fresh_var();
                    param_types.push(param_type.clone());
                    local_env.bind(
                        param.clone(),
                        TypeScheme::mono(param_type, EffectSet::pure_cpu_none()),
                    );
                }

                let (body_type, body_eff) = self.infer_expr(body, &local_env, context_effects)?;

                let func_type = Type::Function(param_types, Box::new(body_type), body_eff.clone());

                Ok((func_type, body_eff))
            }

            Expr::ArrayLit(elems, span) => {
                if elems.is_empty() {
                    let elem_type = self.inferencer.fresh_var();
                    return Ok((
                        Type::Array(Box::new(elem_type), Space::Cpu, ArenaRef::Anonymous),
                        EffectSet::pure_cpu_none().with_allocation(Allocation::Arena),
                    ));
                }

                let (first_type, first_eff) = self.infer_expr(&elems[0], env, context_effects)?;
                let mut merged_effects = first_eff;

                for elem in &elems[1..] {
                    let (elem_type, elem_eff) = self.infer_expr(elem, env, context_effects)?;
                    self.unify_types(&first_type, &elem_type, *span)?;
                    merged_effects = self.effect_checker.merge(&merged_effects, &elem_eff);
                }

                merged_effects.allocation = Allocation::Arena;

                Ok((
                    Type::Array(Box::new(first_type), Space::Cpu, ArenaRef::Anonymous),
                    merged_effects,
                ))
            }

            Expr::Match(scrutinee, arms, span) => {
                let (_, scrutinee_eff) =
                    self.infer_expr(scrutinee, env, context_effects)?;

                if arms.is_empty() {
                    return Err(TypeCheckError::new("Match must have at least one arm", *span));
                }

                let (first_type, first_eff) =
                    self.infer_expr(&arms[0].body, env, context_effects)?;
                let mut merged_effects = self.effect_checker.merge(&scrutinee_eff, &first_eff);

                for arm in &arms[1..] {
                    let (arm_type, arm_eff) = self.infer_expr(&arm.body, env, context_effects)?;
                    self.unify_types(&first_type, &arm_type, *span)?;
                    merged_effects = self.effect_checker.merge(&merged_effects, &arm_eff);
                }

                Ok((first_type, merged_effects))
            }

            Expr::Unsafe(inner, _) => {
                // Unsafe blocks can perform any operation
                // but we still type-check the contents
                self.infer_expr(inner, env, context_effects)
            }

            // Parallel and async primitives
            Expr::ParFor(_, _, _, span) => {
                let effects = EffectSet::pure_cpu_none().with_concurrency(Concurrency::Concurrent);
                self.effect_checker
                    .can_call(context_effects, &effects)
                    .map_err(|e| TypeCheckError::new(e, *span))?;

                Ok((Type::Unit, effects))
            }

            Expr::ParMap(_, array, span) => {
                let effects = EffectSet::pure_cpu_none()
                    .with_allocation(Allocation::Heap)
                    .with_concurrency(Concurrency::Concurrent);

                self.effect_checker
                    .can_call(context_effects, &effects)
                    .map_err(|e| TypeCheckError::new(e, *span))?;

                let (array_type, _) = self.infer_expr(array, env, context_effects)?;

                Ok((array_type, effects))
            }

            Expr::ParMapInplace(_, src, dst, span) => {
                let effects = EffectSet::pure_cpu_none().with_concurrency(Concurrency::Concurrent);

                self.effect_checker
                    .can_call(context_effects, &effects)
                    .map_err(|e| TypeCheckError::new(e, *span))?;

                Ok((Type::Unit, effects))
            }

            Expr::Async(expr, span) => {
                let (inner_type, _) = self.infer_expr(expr, env, context_effects)?;
                let effects = EffectSet::new()
                    .with_purity(Purity::Io)
                    .with_concurrency(Concurrency::Concurrent);

                Ok((Type::Task(Box::new(inner_type)), effects))
            }

            Expr::Await(task, span) => {
                let (task_type, _) = self.infer_expr(task, env, context_effects)?;
                let effects = EffectSet::new().with_purity(Purity::Io);

                match task_type {
                    Type::Task(inner) => Ok((*inner, effects)),
                    _ => Err(TypeCheckError::new("await requires Task type", *span)),
                }
            }

            Expr::Spawn(expr, span) => {
                let (inner_type, _) = self.infer_expr(expr, env, context_effects)?;
                let effects = EffectSet::new()
                    .with_purity(Purity::Io)
                    .with_allocation(Allocation::Heap)
                    .with_concurrency(Concurrency::Concurrent);

                Ok((Type::Actor(Box::new(inner_type)), effects))
            }

            Expr::Send(actor, msg, span) => {
                let effects = EffectSet::new()
                    .with_purity(Purity::Io)
                    .with_concurrency(Concurrency::Concurrent);

                Ok((Type::Unit, effects))
            }

            Expr::Receive(_) => {
                let effects = EffectSet::new()
                    .with_purity(Purity::Io)
                    .with_concurrency(Concurrency::Concurrent);

                let msg_type = self.inferencer.fresh_var();

                Ok((msg_type, effects))
            }

            Expr::NewArena(span) => {
                let effects = EffectSet::pure_cpu_none().with_allocation(Allocation::Arena);

                self.effect_checker
                    .can_call(context_effects, &effects)
                    .map_err(|e| TypeCheckError::new(e, *span))?;

                Ok((Type::Arena, effects))
            }

            Expr::AllocArray(_, _, space, span) => {
                let effects = EffectSet::pure_cpu_none().with_allocation(Allocation::Arena);

                self.effect_checker
                    .can_call(context_effects, &effects)
                    .map_err(|e| TypeCheckError::new(e, *span))?;

                let elem_type = self.inferencer.fresh_var();

                Ok((
                    Type::Array(Box::new(elem_type), space.clone(), ArenaRef::Anonymous),
                    effects,
                ))
            }

            Expr::GpuKernel(expr, _) => {
                let effects = EffectSet::pure_cpu_none().with_execution(Execution::Gpu);

                let (inner_type, _) = self.infer_expr(expr, env, &effects)?;

                Ok((inner_type, effects))
            }

            Expr::CpuToGpu(expr, _) | Expr::GpuToCpu(expr, _) => {
                let effects = EffectSet::new()
                    .with_purity(Purity::Io)
                    .with_allocation(Allocation::Heap);

                let (inner_type, _) = self.infer_expr(expr, env, context_effects)?;

                Ok((inner_type, effects))
            }

            Expr::Log(expr, span) => {
                let effects = EffectSet::new().with_purity(Purity::Debug);

                self.effect_checker
                    .can_call(context_effects, &effects)
                    .map_err(|e| TypeCheckError::new(e, *span))?;

                let (_, _) = self.infer_expr(expr, env, context_effects)?;

                Ok((Type::Unit, effects))
            }

            Expr::Assert(cond, _, span) => {
                let effects = EffectSet::new().with_purity(Purity::Debug);

                self.effect_checker
                    .can_call(context_effects, &effects)
                    .map_err(|e| TypeCheckError::new(e, *span))?;

                let (cond_type, _) = self.infer_expr(cond, env, context_effects)?;
                self.unify_types(&cond_type, &Type::Bool, *span)?;

                Ok((Type::Unit, effects))
            }

            Expr::ArrayIndex(arr, idx, span) => {
                let (arr_type, arr_eff) = self.infer_expr(arr, env, context_effects)?;
                let (idx_type, idx_eff) = self.infer_expr(idx, env, context_effects)?;

                self.unify_types(&idx_type, &Type::Int32, *span)?;

                match arr_type {
                    Type::Array(elem_type, _, _) => {
                        let merged_effects = self.effect_checker.merge(&arr_eff, &idx_eff);
                        Ok((*elem_type, merged_effects))
                    }
                    _ => Err(TypeCheckError::new("Expected array type", *span)),
                }
            }

            Expr::Constructor(name, _, span) => {
                let scheme = env.lookup(name).ok_or_else(|| {
                    TypeCheckError::new(format!("Undefined constructor: {}", name), *span)
                })?;

                Ok((scheme.ty.clone(), scheme.effects.clone()))
            }

            Expr::Seq(exprs, _) => {
                if exprs.is_empty() {
                    return Ok((Type::Unit, EffectSet::pure_cpu_none()));
                }

                let mut merged_effects = EffectSet::pure_cpu_none();
                let mut last_type = Type::Unit;

                for expr in exprs {
                    let (expr_type, expr_eff) = self.infer_expr(expr, env, context_effects)?;
                    last_type = expr_type;
                    merged_effects = self.effect_checker.merge(&merged_effects, &expr_eff);
                }

                Ok((last_type, merged_effects))
            }
        }
    }

    fn unify_types(&mut self, left: &Type, right: &Type, span: Span) -> TypeCheckResult<()> {
        self.inferencer
            .unify(left, right)
            .map_err(|e| TypeCheckError::new(e.message, e.span.unwrap_or(span)))
    }

    fn ensure_integer_like(
        &self,
        ty: &Type,
        span: Span,
        operand_position: &str,
    ) -> TypeCheckResult<()> {
        if ty.is_integer_like() {
            Ok(())
        } else {
            Err(TypeCheckError::new(
                format!(
                    "Expected integer type for {} but found {:?}",
                    operand_position, ty
                ),
                span,
            ))
        }
    }

    pub fn literal_types(&self) -> &HashMap<Span, IntegerKind> {
        &self.literal_types
    }
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}
