use crate::ir::{BinOp, IRFunction, IRInst, IRModule, IRType, IRValue, UnOp};
use anyhow::{anyhow, bail, Result};
use inkwell::basic_block::BasicBlock;
use inkwell::builder::{Builder, BuilderError};
use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::module::Module as LLVMModule;
use inkwell::targets::{InitializationConfig, Target};
use inkwell::types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum};
use inkwell::values::{
    BasicMetadataValueEnum, BasicValue, BasicValueEnum, FunctionValue, PointerValue, ValueKind,
};
use inkwell::{AddressSpace, IntPredicate, OptimizationLevel};
use std::collections::HashMap;

pub struct X86Backend {
    context: &'static Context,
    module: Option<LLVMModule<'static>>,
    execution_engine: Option<ExecutionEngine<'static>>,
}

impl X86Backend {
    pub fn new() -> Result<Self> {
        Target::initialize_native(&InitializationConfig::default())
            .map_err(|e| anyhow!("Failed to initialize native target: {e}"))?;
        let context = Box::leak(Box::new(Context::create()));
        Ok(Self {
            context,
            module: None,
            execution_engine: None,
        })
    }

    pub fn compile(&mut self, ir_module: &IRModule) -> Result<()> {
        let module = self.context.create_module("flux");

        let mut function_map = HashMap::new();
        for func in &ir_module.functions {
            let fn_type = self.build_function_type(func);
            let llvm_fn = module.add_function(&func.name, fn_type, None);
            for (idx, param) in func.params.iter().enumerate() {
                if let Some(value) = llvm_fn.get_nth_param(idx as u32) {
                    value.set_name(&param.name);
                }
            }
            function_map.insert(func.name.clone(), llvm_fn);
        }

        for func in &ir_module.functions {
            let llvm_fn = *function_map
                .get(&func.name)
                .ok_or_else(|| anyhow!("Missing LLVM declaration for {}", func.name))?;
            self.build_function(func, llvm_fn, ir_module, &function_map)?;
        }

        let execution_engine = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .map_err(|e| anyhow!("Failed to create JIT: {e}"))?;

        self.module = Some(module);
        self.execution_engine = Some(execution_engine);
        Ok(())
    }

    pub fn execute_main(&self) -> Result<i32> {
        let engine = self
            .execution_engine
            .as_ref()
            .ok_or_else(|| anyhow!("Backend has not been compiled"))?;

        unsafe {
            let func = engine
                .get_function::<unsafe extern "C" fn() -> i32>("main")
                .map_err(|e| anyhow!("Failed to load main: {e}"))?;
            Ok(func.call())
        }
    }

    fn build_function_type(&self, func: &IRFunction) -> inkwell::types::FunctionType<'static> {
        let params: Vec<BasicMetadataTypeEnum> = func
            .params
            .iter()
            .map(|p| self.basic_type(&p.ty).into())
            .collect();

        match func.return_type {
            IRType::Void => self.context.void_type().fn_type(&params, false),
            _ => self.basic_type(&func.return_type).fn_type(&params, false),
        }
    }

    fn build_function(
        &self,
        ir_func: &IRFunction,
        llvm_fn: FunctionValue<'static>,
        ir_module: &IRModule,
        functions: &HashMap<String, FunctionValue<'static>>,
    ) -> Result<()> {
        let builder = self.context.create_builder();
        let entry_builder = self.context.create_builder();
        let entry_block = self.context.append_basic_block(llvm_fn, "entry");
        builder.position_at_end(entry_block);
        entry_builder.position_at_end(entry_block);

        let mut var_ptrs: HashMap<String, PointerValue<'static>> = HashMap::new();
        let mut var_types: HashMap<String, IRType> = HashMap::new();

        for (idx, param) in ir_func.params.iter().enumerate() {
            let llvm_param = llvm_fn
                .get_nth_param(idx as u32)
                .ok_or_else(|| anyhow!("Missing LLVM param {}", param.name))?;
            let alloca = map_build(entry_builder.build_alloca(
                self.basic_type(&param.ty),
                &format!("{}.addr", param.name),
            ))?;
            map_build(builder.build_store(alloca, llvm_param))?;
            var_ptrs.insert(param.name.clone(), alloca);
            var_types.insert(param.name.clone(), param.ty.clone());
        }

        let mut label_blocks: HashMap<String, BasicBlock<'static>> = HashMap::new();
        let mut first_label_seen = false;
        for inst in &ir_func.body {
            if let IRInst::Label(name) = inst {
                if label_blocks.contains_key(name) {
                    continue;
                }
                let block = if !first_label_seen {
                    first_label_seen = true;
                    entry_block
                } else {
                    self.context.append_basic_block(llvm_fn, name)
                };
                label_blocks.insert(name.clone(), block);
            }
        }

        for inst in &ir_func.body {
            match inst {
                IRInst::Assign(dest, value) => {
                    let value_ty = self.value_type(value, &var_types)?;
                    let llvm_value = self.build_value(value, &builder, &var_ptrs, &var_types)?;
                    self.store_var(
                        dest,
                        &value_ty,
                        llvm_value,
                        &entry_builder,
                        &builder,
                        &mut var_ptrs,
                    )?;
                    var_types.insert(dest.clone(), value_ty);
                }
                IRInst::BinOp(dest, op, left, right) => {
                    let left_ty = self.value_type(left, &var_types)?;
                    let right_ty = self.value_type(right, &var_types)?;
                    if left_ty != right_ty {
                        bail!("Type mismatch in binop: {:?} vs {:?}", left_ty, right_ty);
                    }
                    let left_val = self.build_value(left, &builder, &var_ptrs, &var_types)?;
                    let right_val = self.build_value(right, &builder, &var_ptrs, &var_types)?;
                    let result_val =
                        self.build_binop(op, &left_ty, left_val, right_val, &builder)?;
                    let result_ty = match op {
                        BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                            IRType::Bool
                        }
                        BinOp::And | BinOp::Or => IRType::Bool,
                        _ => left_ty.clone(),
                    };
                    self.store_var(
                        dest,
                        &result_ty,
                        result_val,
                        &entry_builder,
                        &builder,
                        &mut var_ptrs,
                    )?;
                    var_types.insert(dest.clone(), result_ty);
                }
                IRInst::UnOp(dest, op, operand) => {
                    let operand_ty = self.value_type(operand, &var_types)?;
                    let operand_val =
                        self.build_value(operand, &builder, &var_ptrs, &var_types)?;
                    let result_val = self.build_unop(op, &operand_ty, operand_val, &builder)?;
                    let result_ty = match op {
                        UnOp::Not => IRType::Bool,
                        UnOp::Neg => operand_ty.clone(),
                    };
                    self.store_var(
                        dest,
                        &result_ty,
                        result_val,
                        &entry_builder,
                        &builder,
                        &mut var_ptrs,
                    )?;
                    var_types.insert(dest.clone(), result_ty);
                }
                IRInst::Label(name) => {
                    if let Some(block) = label_blocks.get(name) {
                        builder.position_at_end(*block);
                    }
                }
                IRInst::Jump(target) => {
                    let dest_block = label_blocks
                        .get(target)
                        .copied()
                        .ok_or_else(|| anyhow!("Unknown label {target}"))?;
                    map_build(builder.build_unconditional_branch(dest_block))?;
                }
                IRInst::CondJump(cond, true_label, false_label) => {
                    let cond_val = self
                        .build_value(cond, &builder, &var_ptrs, &var_types)?
                        .into_int_value();
                    let true_block = label_blocks
                        .get(true_label)
                        .copied()
                        .ok_or_else(|| anyhow!("Unknown label {true_label}"))?;
                    let false_block = label_blocks
                        .get(false_label)
                        .copied()
                        .ok_or_else(|| anyhow!("Unknown label {false_label}"))?;
                    map_build(builder.build_conditional_branch(
                        cond_val,
                        true_block,
                        false_block,
                    ))?;
                }
                IRInst::Return(value) => {
                    if let Some(val) = value {
                        let llvm_val =
                            self.build_value(val, &builder, &var_ptrs, &var_types)?;
                        map_build(builder.build_return(Some(&llvm_val)))?;
                    } else {
                        map_build(builder.build_return(None))?;
                    }
                }
                IRInst::Call(dest, name, args) => {
                    let callee = functions
                        .get(name)
                        .copied()
                        .ok_or_else(|| anyhow!("Unknown function {name}"))?;
                    let mut llvm_args = Vec::with_capacity(args.len());
                    for arg in args {
                        let val = self.build_value(arg, &builder, &var_ptrs, &var_types)?;
                        llvm_args.push(BasicMetadataValueEnum::from(val));
                    }
                    let call_site = map_build(builder.build_call(
                        callee,
                        &llvm_args,
                        dest.as_deref().unwrap_or("calltmp"),
                    ))?;
                    if let Some(dest_name) = dest {
                        if let ValueKind::Basic(ret_val) = call_site.try_as_basic_value() {
                            let ret_ty = self.return_type(name, ir_module, functions)?;
                            self.store_var(
                                dest_name,
                                &ret_ty,
                                ret_val,
                                &entry_builder,
                                &builder,
                                &mut var_ptrs,
                            )?;
                            var_types.insert(dest_name.clone(), ret_ty);
                        }
                    }
                }
                _ => {
                    bail!("Unsupported IR instruction in LLVM backend: {inst:?}");
                }
            }
        }

        Ok(())
    }

    fn return_type(
        &self,
        name: &str,
        ir_module: &IRModule,
        functions: &HashMap<String, FunctionValue<'static>>,
    ) -> Result<IRType> {
        if let Some(func) = functions.get(name) {
            if let Some(ret) = func.get_type().get_return_type() {
                return Ok(self.ir_type_from_basic(ret));
            } else {
                return Ok(IRType::Void);
            }
        }

        if let Some(func) = ir_module.find_function(name) {
            return Ok(func.return_type.clone());
        }

        Err(anyhow!("Unable to determine return type for {name}"))
    }

    fn ir_type_from_basic(&self, ty: BasicTypeEnum<'static>) -> IRType {
        match ty {
            BasicTypeEnum::IntType(int_ty) => match int_ty.get_bit_width() {
                1 => IRType::Bool,
                32 => IRType::I32,
                64 => IRType::I64,
                _ => IRType::I32,
            },
            BasicTypeEnum::FloatType(_) => IRType::F32,
            BasicTypeEnum::PointerType(_) => IRType::Ptr,
            BasicTypeEnum::StructType(struct_ty) => {
                let field_types = struct_ty
                    .get_field_types()
                    .into_iter()
                    .map(|field| self.ir_type_from_basic(field.into()))
                    .collect();
                IRType::Struct(field_types)
            }
            BasicTypeEnum::ArrayType(_) => IRType::Ptr,
            BasicTypeEnum::VectorType(_) => IRType::Ptr,
            BasicTypeEnum::ScalableVectorType(_) => IRType::Ptr,
        }
    }

    fn store_var(
        &self,
        name: &str,
        ty: &IRType,
        value: BasicValueEnum<'static>,
        entry_builder: &Builder<'static>,
        current_builder: &Builder<'static>,
        vars: &mut HashMap<String, PointerValue<'static>>,
    ) -> Result<()> {
        let ptr = if let Some(ptr) = vars.get(name) {
            *ptr
        } else {
            let alloca = map_build(entry_builder.build_alloca(
                self.basic_type(ty),
                &format!("{}.addr", name),
            ))?;
            vars.insert(name.to_string(), alloca);
            alloca
        };
        map_build(current_builder.build_store(ptr, value))?;
        Ok(())
    }

    fn build_binop(
        &self,
        op: &BinOp,
        ty: &IRType,
        left: BasicValueEnum<'static>,
        right: BasicValueEnum<'static>,
        builder: &Builder<'static>,
    ) -> Result<BasicValueEnum<'static>> {
        match ty {
            IRType::I32 | IRType::I64 => {
                let l = left.into_int_value();
                let r = right.into_int_value();
                let result = match op {
                    BinOp::Add => map_build(builder.build_int_add(l, r, "addtmp"))?,
                    BinOp::Sub => map_build(builder.build_int_sub(l, r, "subtmp"))?,
                    BinOp::Mul => map_build(builder.build_int_mul(l, r, "multmp"))?,
                    BinOp::Div => map_build(builder.build_int_signed_div(l, r, "divtmp"))?,
                    BinOp::Mod => map_build(builder.build_int_signed_rem(l, r, "remtmp"))?,
                    BinOp::Eq => map_build(builder.build_int_compare(IntPredicate::EQ, l, r, "eqtmp"))?,
                    BinOp::Ne => map_build(builder.build_int_compare(IntPredicate::NE, l, r, "netmp"))?,
                    BinOp::Lt => map_build(builder.build_int_compare(IntPredicate::SLT, l, r, "lttmp"))?,
                    BinOp::Le => map_build(builder.build_int_compare(IntPredicate::SLE, l, r, "letmp"))?,
                    BinOp::Gt => map_build(builder.build_int_compare(IntPredicate::SGT, l, r, "gttmp"))?,
                    BinOp::Ge => map_build(builder.build_int_compare(IntPredicate::SGE, l, r, "getmp"))?,
                    BinOp::And => map_build(builder.build_and(l, r, "andtmp"))?,
                    BinOp::Or => map_build(builder.build_or(l, r, "ortmp"))?,
                };
                Ok(result.as_basic_value_enum())
            }
            IRType::Bool => {
                let l = left.into_int_value();
                let r = right.into_int_value();
                let result = match op {
                    BinOp::And => map_build(builder.build_and(l, r, "andbool"))?,
                    BinOp::Or => map_build(builder.build_or(l, r, "orbool"))?,
                    BinOp::Eq => map_build(builder.build_int_compare(IntPredicate::EQ, l, r, "eqbool"))?,
                    BinOp::Ne => map_build(builder.build_int_compare(IntPredicate::NE, l, r, "nebool"))?,
                    _ => bail!("Unsupported bool binop: {op:?}"),
                };
                Ok(result.as_basic_value_enum())
            }
            _ => bail!("Unsupported binop type: {ty:?}"),
        }
    }

    fn build_unop(
        &self,
        op: &UnOp,
        ty: &IRType,
        operand: BasicValueEnum<'static>,
        builder: &Builder<'static>,
    ) -> Result<BasicValueEnum<'static>> {
        match op {
            UnOp::Neg => match ty {
                IRType::I32 | IRType::I64 => {
                    let zero = operand.into_int_value().get_type().const_zero();
                    let result = map_build(builder.build_int_sub(
                        zero,
                        operand.into_int_value(),
                        "negtmp",
                    ))?;
                    Ok(result.as_basic_value_enum())
                }
                _ => bail!("Unsupported negation type: {ty:?}"),
            },
            UnOp::Not => {
                let val = operand.into_int_value();
                Ok(map_build(builder.build_not(val, "nottmp"))?.as_basic_value_enum())
            }
        }
    }

    fn build_value(
        &self,
        value: &IRValue,
        builder: &Builder<'static>,
        vars: &HashMap<String, PointerValue<'static>>,
        types: &HashMap<String, IRType>,
    ) -> Result<BasicValueEnum<'static>> {
        let val = match value {
            IRValue::I32(n) => self.context.i32_type().const_int(*n as u64, true).into(),
            IRValue::I64(n) => self.context.i64_type().const_int(*n as u64, true).into(),
            IRValue::Bool(b) => self.context.bool_type().const_int(*b as u64, false).into(),
            IRValue::Null => self
                .context
                .ptr_type(AddressSpace::from(0u16))
                .const_null()
                .into(),
            IRValue::Var(name) => {
                let ptr = vars
                    .get(name)
                    .copied()
                    .ok_or_else(|| anyhow!("Use of uninitialized variable {name}"))?;
                let ty = types
                    .get(name)
                    .ok_or_else(|| anyhow!("Unknown variable type for {name}"))?;
                map_build(builder.build_load(
                    self.basic_type(ty),
                    ptr,
                    &format!("load_{name}"),
                ))?
            }
            other => bail!("Unsupported IR value: {other:?}"),
        };
        Ok(val)
    }

    fn value_type(
        &self,
        value: &IRValue,
        vars: &HashMap<String, IRType>,
    ) -> Result<IRType> {
        Ok(match value {
            IRValue::I32(_) => IRType::I32,
            IRValue::I64(_) => IRType::I64,
            IRValue::Bool(_) => IRType::Bool,
            IRValue::Var(name) => vars
                .get(name)
                .cloned()
                .ok_or_else(|| anyhow!("Unknown variable type for {name}"))?,
            IRValue::Null => IRType::Ptr,
            other => bail!("Unsupported value type inference for {other:?}"),
        })
    }

    fn basic_type(&self, ty: &IRType) -> BasicTypeEnum<'static> {
        match ty {
            IRType::I32 => self.context.i32_type().into(),
            IRType::I64 => self.context.i64_type().into(),
            IRType::F32 => self.context.f32_type().into(),
            IRType::F64 => self.context.f64_type().into(),
            IRType::Bool => self.context.bool_type().into(),
            IRType::Ptr => self.context.ptr_type(AddressSpace::from(0u16)).into(),
            IRType::Struct(fields) => {
                let field_types: Vec<_> = fields.iter().map(|f| self.basic_type(f)).collect();
                self.context.struct_type(&field_types, false).into()
            }
            IRType::Void => self.context.i8_type().into(),
        }
    }
}

impl Default for X86Backend {
    fn default() -> Self {
        Self::new().expect("Failed to initialize LLVM backend")
    }
}

fn map_build<T>(result: Result<T, BuilderError>) -> Result<T> {
    result.map_err(|e| anyhow!("LLVM build error: {e:?}"))
}
