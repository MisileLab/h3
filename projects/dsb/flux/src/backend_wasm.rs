use crate::ir::{IRModule, IRFunction, IRInst, IRValue, IRType};
use anyhow::Result;
use wasm_encoder::*;
use std::collections::HashMap;

pub struct WasmBackend {
    module: Module,
    type_section: TypeSection,
    function_section: FunctionSection,
    export_section: ExportSection,
    code_section: CodeSection,
    func_types: Vec<u32>,
}

impl WasmBackend {
    pub fn new() -> Self {
        WasmBackend {
            module: Module::new(),
            type_section: TypeSection::new(),
            function_section: FunctionSection::new(),
            export_section: ExportSection::new(),
            code_section: CodeSection::new(),
            func_types: Vec::new(),
        }
    }

    pub fn compile(&mut self, ir_module: &IRModule) -> Result<Vec<u8>> {
        let mut type_indices = HashMap::new();

        // Create type signatures
        for func in ir_module.functions.iter() {
            let params: Vec<ValType> = func
                .params
                .iter()
                .map(|p| self.ir_type_to_wasm(&p.ty))
                .collect();

            let results = if func.return_type != IRType::Void {
                vec![self.ir_type_to_wasm(&func.return_type)]
            } else {
                vec![]
            };

            let type_idx = self.type_section.len();
            self.type_section.ty().function(params, results);
            type_indices.insert(func.name.clone(), type_idx);
        }

        // Declare functions
        for func in &ir_module.functions {
            let type_idx = type_indices[&func.name];
            self.function_section.function(type_idx);
            self.func_types.push(type_idx);
        }

        // Generate code for each function
        for func in &ir_module.functions {
            self.compile_function(func)?;
        }

        // Export functions
        for (idx, func) in ir_module.functions.iter().enumerate() {
            if func.is_export {
                self.export_section
                    .export(&func.name, ExportKind::Func, idx as u32);
            }
        }

        // Build module by consuming sections
        let type_section = std::mem::take(&mut self.type_section);
        let function_section = std::mem::take(&mut self.function_section);
        let export_section = std::mem::take(&mut self.export_section);
        let code_section = std::mem::take(&mut self.code_section);

        self.module.section(&type_section);
        self.module.section(&function_section);
        self.module.section(&export_section);
        self.module.section(&code_section);

        let module = std::mem::replace(&mut self.module, Module::new());
        Ok(module.finish())
    }

    fn compile_function(&mut self, ir_func: &IRFunction) -> Result<()> {
        let mut vars = HashMap::new();
        let mut locals_vec = Vec::new();

        // Parameters are variables 0..n
        for (i, param) in ir_func.params.iter().enumerate() {
            vars.insert(param.name.clone(), i as u32);
        }
        let mut var_counter = ir_func.params.len() as u32;

        // Generate locals for temporaries
        let mut locals_map = HashMap::new();

        // First pass: collect all variable declarations
        for inst in &ir_func.body {
            match inst {
                IRInst::Assign(dest, _) | IRInst::BinOp(dest, _, _, _) | IRInst::UnOp(dest, _, _) => {
                    if !vars.contains_key(dest) && !locals_map.contains_key(dest) {
                        locals_map.insert(dest.clone(), var_counter);
                        var_counter += 1;
                    }
                }
                IRInst::Call(Some(dest), _, _) => {
                    if !vars.contains_key(dest) && !locals_map.contains_key(dest) {
                        locals_map.insert(dest.clone(), var_counter);
                        var_counter += 1;
                    }
                }
                _ => {}
            }
        }

        // Merge locals_map into vars
        for (name, idx) in locals_map {
            vars.insert(name, idx);
        }

        // Declare locals (all as i32 for simplicity)
        let num_locals = var_counter - ir_func.params.len() as u32;
        if num_locals > 0 {
            locals_vec.push((num_locals, ValType::I32));
        }

        // Build function body
        let mut func_body = Function::new(locals_vec);

        // Generate code
        for inst in &ir_func.body {
            self.generate_instruction(&mut func_body, inst, &vars)?;
        }

        // Ensure function ends with return or end
        func_body.instruction(&Instruction::End);

        self.code_section.function(&func_body);

        Ok(())
    }

    fn generate_instruction(
        &self,
        func: &mut Function,
        inst: &IRInst,
        vars: &HashMap<String, u32>,
    ) -> Result<()> {
        match inst {
            IRInst::Assign(dest, value) => {
                self.generate_value(func, value, vars)?;
                let var_idx = vars[dest];
                func.instruction(&Instruction::LocalSet(var_idx));
            }

            IRInst::BinOp(dest, op, left, right) => {
                self.generate_value(func, left, vars)?;
                self.generate_value(func, right, vars)?;

                match op {
                    crate::ir::BinOp::Add => func.instruction(&Instruction::I32Add),
                    crate::ir::BinOp::Sub => func.instruction(&Instruction::I32Sub),
                    crate::ir::BinOp::Mul => func.instruction(&Instruction::I32Mul),
                    crate::ir::BinOp::Div => func.instruction(&Instruction::I32DivS),
                    crate::ir::BinOp::Mod => func.instruction(&Instruction::I32RemS),
                    crate::ir::BinOp::Eq => func.instruction(&Instruction::I32Eq),
                    crate::ir::BinOp::Ne => func.instruction(&Instruction::I32Ne),
                    crate::ir::BinOp::Lt => func.instruction(&Instruction::I32LtS),
                    crate::ir::BinOp::Le => func.instruction(&Instruction::I32LeS),
                    crate::ir::BinOp::Gt => func.instruction(&Instruction::I32GtS),
                    crate::ir::BinOp::Ge => func.instruction(&Instruction::I32GeS),
                    crate::ir::BinOp::And => func.instruction(&Instruction::I32And),
                    crate::ir::BinOp::Or => func.instruction(&Instruction::I32Or),
                };

                let var_idx = vars[dest];
                func.instruction(&Instruction::LocalSet(var_idx));
            }

            IRInst::UnOp(dest, op, operand) => {
                self.generate_value(func, operand, vars)?;

                match op {
                    crate::ir::UnOp::Neg => {
                        func.instruction(&Instruction::I32Const(0));
                        func.instruction(&Instruction::I32Sub);
                    }
                    crate::ir::UnOp::Not => {
                        func.instruction(&Instruction::I32Eqz);
                    }
                };

                let var_idx = vars[dest];
                func.instruction(&Instruction::LocalSet(var_idx));
            }

            IRInst::Return(value) => {
                if let Some(val) = value {
                    self.generate_value(func, val, vars)?;
                }
                func.instruction(&Instruction::Return);
            }

            IRInst::CondJump(cond, _, _) => {
                // WASM control flow is different - simplified for now
                self.generate_value(func, cond, vars)?;
                func.instruction(&Instruction::If(BlockType::Empty));
                // Would need to properly handle blocks here
                func.instruction(&Instruction::End);
            }

            IRInst::Label(_) | IRInst::Jump(_) => {
                // WASM uses structured control flow, not labels
                // In a full implementation, we'd translate to if/block/loop
            }

            IRInst::Call(dest, _func_name, args) => {
                // Generate arguments
                for arg in args {
                    self.generate_value(func, arg, vars)?;
                }

                // Simplified: just push a dummy value for now
                if let Some(dest) = dest {
                    func.instruction(&Instruction::I32Const(0));
                    let var_idx = vars[dest];
                    func.instruction(&Instruction::LocalSet(var_idx));
                }
            }

            _ => {}
        }

        Ok(())
    }

    fn generate_value(
        &self,
        func: &mut Function,
        value: &IRValue,
        vars: &HashMap<String, u32>,
    ) -> Result<()> {
        match value {
            IRValue::I32(n) => {
                func.instruction(&Instruction::I32Const(*n));
            }
            IRValue::I64(n) => {
                func.instruction(&Instruction::I64Const(*n));
            }
            IRValue::Bool(b) => {
                func.instruction(&Instruction::I32Const(if *b { 1 } else { 0 }));
            }
            IRValue::Var(name) => {
                let var_idx = vars
                    .get(name)
                    .ok_or_else(|| anyhow::anyhow!("Undefined variable: {}", name))?;
                func.instruction(&Instruction::LocalGet(*var_idx));
            }
            _ => {
                func.instruction(&Instruction::I32Const(0));
            }
        }

        Ok(())
    }

    fn ir_type_to_wasm(&self, ty: &IRType) -> ValType {
        match ty {
            IRType::I32 | IRType::Bool => ValType::I32,
            IRType::I64 | IRType::Ptr => ValType::I64,
            IRType::F32 => ValType::F32,
            IRType::F64 => ValType::F64,
            _ => ValType::I32,
        }
    }

    pub fn write_to_file(&self, bytes: &[u8], path: &str) -> Result<()> {
        std::fs::write(path, bytes)?;
        Ok(())
    }
}

impl Default for WasmBackend {
    fn default() -> Self {
        Self::new()
    }
}
