use crate::ast::{Module, Declaration, FunctionDecl, Expr, BinOp as AstBinOp, UnOp as AstUnOp, Type, IntegerKind, Span};
use crate::ir::{IRModule, IRParam, IRType, IRValue, IRBuilder, BinOp as IrBinOp, UnOp as IrUnOp};
use std::collections::HashMap;

pub struct CodeGenerator {
    builder: IRBuilder,
    module: IRModule,
    locals: HashMap<String, IRType>,
    literal_types: HashMap<Span, IntegerKind>,
}

impl CodeGenerator {
    pub fn new() -> Self {
        CodeGenerator {
            builder: IRBuilder::new(),
            module: IRModule::new(),
            locals: HashMap::new(),
            literal_types: HashMap::new(),
        }
    }

    pub fn set_literal_types(&mut self, literal_types: HashMap<Span, IntegerKind>) {
        self.literal_types = literal_types;
    }

    pub fn generate(&mut self, ast_module: &Module) -> Result<IRModule, String> {
        for decl in &ast_module.declarations {
            match decl {
                Declaration::Function(func) => {
                    self.generate_function(func)?;
                }
                _ => {}
            }
        }

        Ok(self.module.clone())
    }

    fn generate_function(&mut self, func: &FunctionDecl) -> Result<(), String> {
        self.locals.clear();

        let params: Vec<IRParam> = func
            .params
            .iter()
            .map(|p| {
                let ty = IRType::I32; // Simplified
                self.locals.insert(p.clone(), ty.clone());
                IRParam {
                    name: p.clone(),
                    ty,
                }
            })
            .collect();

        let return_type = self.type_to_ir(
            func.type_sig
                .as_ref()
                .map(|sig| &*sig.return_type)
                .unwrap_or(&Type::Unit),
        );

        self.builder
            .start_function(func.name.clone(), params, return_type.clone());

        let result = self.generate_expr(&func.body)?;

        // Add return
        if return_type != IRType::Void {
            self.builder.return_value(Some(result));
        } else {
            self.builder.return_value(None);
        }

        let ir_func = self
            .builder
            .finish_function()
            .ok_or("No function being built")?;

        // Mark main as export
        let ir_func = if func.name == "main" {
            ir_func.make_export()
        } else {
            ir_func
        };

        self.module.add_function(ir_func);

        Ok(())
    }

    fn generate_expr(&mut self, expr: &Expr) -> Result<IRValue, String> {
        match expr {
            Expr::IntLit(n, span) => self.lower_int_literal(*n, *span),

            Expr::FloatLit(f, _) => Ok(IRValue::F64(*f)),

            Expr::BoolLit(b, _) => Ok(IRValue::Bool(*b)),

            Expr::Unit(_) => Ok(IRValue::I32(0)),

            Expr::Var(name, _) => Ok(IRValue::Var(name.clone())),

            Expr::BinOp(op, left, right, _) => {
                let left_val = self.generate_expr(left)?;
                let right_val = self.generate_expr(right)?;

                let temp = self.builder.fresh_temp();

                let ir_op = match op {
                    AstBinOp::Add => IrBinOp::Add,
                    AstBinOp::Sub => IrBinOp::Sub,
                    AstBinOp::Mul => IrBinOp::Mul,
                    AstBinOp::Div => IrBinOp::Div,
                    AstBinOp::Mod => IrBinOp::Mod,
                    AstBinOp::Eq => IrBinOp::Eq,
                    AstBinOp::Ne => IrBinOp::Ne,
                    AstBinOp::Lt => IrBinOp::Lt,
                    AstBinOp::Le => IrBinOp::Le,
                    AstBinOp::Gt => IrBinOp::Gt,
                    AstBinOp::Ge => IrBinOp::Ge,
                    AstBinOp::And => IrBinOp::And,
                    AstBinOp::Or => IrBinOp::Or,
                };

                self.builder.binop(temp.clone(), ir_op, left_val, right_val);

                Ok(IRValue::Var(temp))
            }

            Expr::UnOp(op, operand, _) => {
                let operand_val = self.generate_expr(operand)?;

                let temp = self.builder.fresh_temp();

                let ir_op = match op {
                    AstUnOp::Neg => IrUnOp::Neg,
                    AstUnOp::Not => IrUnOp::Not,
                };

                self.builder.unop(temp.clone(), ir_op, operand_val);

                Ok(IRValue::Var(temp))
            }

            Expr::If(cond, then_br, else_br, _) => {
                let cond_val = self.generate_expr(cond)?;

                let then_label = self.builder.fresh_label();
                let else_label = self.builder.fresh_label();
                let end_label = self.builder.fresh_label();

                self.builder
                    .cond_jump(cond_val, then_label.clone(), else_label.clone());

                // Then branch
                self.builder.label(then_label);
                let then_val = self.generate_expr(then_br)?;
                let result_temp = self.builder.fresh_temp();
                self.builder.assign(result_temp.clone(), then_val);
                self.builder.jump(end_label.clone());

                // Else branch
                self.builder.label(else_label);
                let else_val = self.generate_expr(else_br)?;
                self.builder.assign(result_temp.clone(), else_val);
                self.builder.jump(end_label.clone());

                // End
                self.builder.label(end_label);

                Ok(IRValue::Var(result_temp))
            }

            Expr::Let(name, value, body, _) => {
                let value_val = self.generate_expr(value)?;
                self.builder.assign(name.clone(), value_val);
                self.locals.insert(name.clone(), IRType::I32);

                self.generate_expr(body)
            }

            Expr::App(func, arg, _) => {
                // Simplified: assume direct function call
                let arg_val = self.generate_expr(arg)?;

                if let Expr::Var(func_name, _) = &**func {
                    let result_temp = self.builder.fresh_temp();
                    self.builder.call(
                        Some(result_temp.clone()),
                        func_name.clone(),
                        vec![arg_val],
                    );
                    Ok(IRValue::Var(result_temp))
                } else {
                    Err("Only direct function calls supported in IR generation".to_string())
                }
            }

            Expr::ParFor(start, end, _, _) => {
                // Generate a call to runtime par_for
                let start_val = self.generate_expr(start)?;
                let end_val = self.generate_expr(end)?;

                self.builder.call(
                    None,
                    "runtime_par_for".to_string(),
                    vec![start_val, end_val],
                );

                Ok(IRValue::I32(0))
            }

            Expr::Async(_, _) => {
                // Generate call to async runtime
                let result_temp = self.builder.fresh_temp();
                self.builder
                    .call(Some(result_temp.clone()), "runtime_async".to_string(), vec![]);
                Ok(IRValue::Var(result_temp))
            }

            Expr::Await(task, _) => {
                let task_val = self.generate_expr(task)?;
                let result_temp = self.builder.fresh_temp();
                self.builder.call(
                    Some(result_temp.clone()),
                    "runtime_await".to_string(),
                    vec![task_val],
                );
                Ok(IRValue::Var(result_temp))
            }

            Expr::Log(expr, _) => {
                let val = self.generate_expr(expr)?;
                self.builder
                    .call(None, "runtime_log".to_string(), vec![val]);
                Ok(IRValue::I32(0))
            }

            Expr::NewArena(_) => {
                let result_temp = self.builder.fresh_temp();
                self.builder
                    .call(Some(result_temp.clone()), "runtime_new_arena".to_string(), vec![]);
                Ok(IRValue::Var(result_temp))
            }

            // Simplified handling for other cases
            _ => Ok(IRValue::I32(0)),
        }
    }

    fn lower_int_literal(&self, value: i128, span: Span) -> Result<IRValue, String> {
        let kind = self
            .literal_types
            .get(&span)
            .copied()
            .unwrap_or(IntegerKind::I32);

        match kind {
            IntegerKind::I8 | IntegerKind::I16 | IntegerKind::I32 => {
                Ok(IRValue::I32(value as i32))
            }
            IntegerKind::U8 | IntegerKind::U16 => Ok(IRValue::I32(value as i32)),
            IntegerKind::U32 => Ok(IRValue::I64(value as i64)),
            IntegerKind::I64 => Ok(IRValue::I64(value as i64)),
            IntegerKind::U64 => {
                if value < 0 {
                    return Err(format!("u64 literal cannot be negative: {}", value));
                }
                if value > i64::MAX as i128 {
                    return Err(format!(
                        "u64 literal {} exceeds current backend support (max {})",
                        value,
                        i64::MAX
                    ));
                }
                Ok(IRValue::I64(value as i64))
            }
        }
    }

    fn type_to_ir(&self, ty: &Type) -> IRType {
        match ty {
            Type::Int32 => IRType::I32,
            Type::Int64 => IRType::I64,
            Type::Float32 => IRType::F32,
            Type::Float64 => IRType::F64,
            Type::Bool => IRType::Bool,
            Type::Unit => IRType::Void,
            _ => IRType::I32, // Simplified
        }
    }
}

impl Default for CodeGenerator {
    fn default() -> Self {
        Self::new()
    }
}
