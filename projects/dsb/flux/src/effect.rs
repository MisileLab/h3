use crate::ast::*;

/// Effect checker - validates that effect sets follow the subset rules
pub struct EffectChecker;

impl EffectChecker {
    pub fn new() -> Self {
        EffectChecker
    }

    /// Check if `caller_effects` can call something with `callee_effects`
    /// Returns Ok(()) if valid, Err with message if not
    pub fn can_call(
        &self,
        caller_effects: &EffectSet,
        callee_effects: &EffectSet,
    ) -> Result<(), String> {
        // Check purity: caller must have equal or more permissive purity
        if !self.purity_allows(caller_effects.purity, callee_effects.purity) {
            return Err(format!(
                "Cannot call function with {:?} effect from {:?} context",
                callee_effects.purity, caller_effects.purity
            ));
        }

        // Check execution space: must match
        if caller_effects.execution != callee_effects.execution {
            // Allow some flexibility: CPU can call CPU, GPU can call GPU
            // Cross-boundary calls require explicit marshaling
            if !self.execution_compatible(caller_effects.execution, callee_effects.execution) {
                return Err(format!(
                    "Cannot call {:?} function from {:?} context",
                    callee_effects.execution, caller_effects.execution
                ));
            }
        }

        // Check allocation: caller must have equal or more permissive allocation
        if !self.allocation_allows(caller_effects.allocation, callee_effects.allocation) {
            return Err(format!(
                "Cannot call function with {:?} allocation from {:?} context",
                callee_effects.allocation, caller_effects.allocation
            ));
        }

        // Check concurrency: caller must have equal or more permissive concurrency
        if !self.concurrency_allows(caller_effects.concurrency, callee_effects.concurrency) {
            return Err(format!(
                "Cannot call function with {:?} concurrency from {:?} context",
                callee_effects.concurrency, caller_effects.concurrency
            ));
        }

        Ok(())
    }

    /// Check if caller purity allows callee purity
    fn purity_allows(&self, caller: Purity, callee: Purity) -> bool {
        match (caller, callee) {
            // Pure can only call pure
            (Purity::Pure, Purity::Pure) => true,
            (Purity::Pure, _) => false,

            // Debug can call pure and debug
            (Purity::Debug, Purity::Pure) => true,
            (Purity::Debug, Purity::Debug) => true,
            (Purity::Debug, _) => false,

            // State can call pure, debug, and state
            (Purity::State, Purity::Pure) => true,
            (Purity::State, Purity::Debug) => true,
            (Purity::State, Purity::State) => true,
            (Purity::State, Purity::Io) => false,

            // IO can call anything
            (Purity::Io, _) => true,
        }
    }

    /// Check if execution spaces are compatible
    fn execution_compatible(&self, caller: Execution, callee: Execution) -> bool {
        match (caller, callee) {
            // Same execution space is always OK
            (Execution::Cpu, Execution::Cpu) => true,
            (Execution::Gpu, Execution::Gpu) => true,

            // CPU can call GPU functions (with explicit marshaling in practice)
            (Execution::Cpu, Execution::Gpu) => true,

            // GPU cannot directly call CPU functions
            (Execution::Gpu, Execution::Cpu) => false,
        }
    }

    /// Check if caller allocation allows callee allocation
    fn allocation_allows(&self, caller: Allocation, callee: Allocation) -> bool {
        match (caller, callee) {
            // None can only call none
            (Allocation::None, Allocation::None) => true,
            (Allocation::None, _) => false,

            // Arena can call none and arena
            (Allocation::Arena, Allocation::None) => true,
            (Allocation::Arena, Allocation::Arena) => true,
            (Allocation::Arena, Allocation::Heap) => false,

            // Heap can call anything
            (Allocation::Heap, _) => true,
        }
    }

    /// Check if caller concurrency allows callee concurrency
    fn concurrency_allows(&self, caller: Concurrency, callee: Concurrency) -> bool {
        match (caller, callee) {
            // Single can only call single
            (Concurrency::Single, Concurrency::Single) => true,
            (Concurrency::Single, Concurrency::Concurrent) => false,

            // Concurrent can call anything
            (Concurrency::Concurrent, _) => true,
        }
    }

    /// Merge two effect sets (take the more permissive effect for each dimension)
    pub fn merge(&self, e1: &EffectSet, e2: &EffectSet) -> EffectSet {
        EffectSet {
            purity: self.merge_purity(e1.purity, e2.purity),
            execution: self.merge_execution(e1.execution, e2.execution),
            allocation: self.merge_allocation(e1.allocation, e2.allocation),
            concurrency: self.merge_concurrency(e1.concurrency, e2.concurrency),
        }
    }

    fn merge_purity(&self, p1: Purity, p2: Purity) -> Purity {
        match (p1, p2) {
            (Purity::Io, _) | (_, Purity::Io) => Purity::Io,
            (Purity::State, _) | (_, Purity::State) => Purity::State,
            (Purity::Debug, _) | (_, Purity::Debug) => Purity::Debug,
            (Purity::Pure, Purity::Pure) => Purity::Pure,
        }
    }

    fn merge_execution(&self, e1: Execution, e2: Execution) -> Execution {
        // If there's any mismatch, we have a problem - but for merging,
        // we'll prefer GPU (more restrictive)
        match (e1, e2) {
            (Execution::Gpu, _) | (_, Execution::Gpu) => Execution::Gpu,
            _ => Execution::Cpu,
        }
    }

    fn merge_allocation(&self, a1: Allocation, a2: Allocation) -> Allocation {
        match (a1, a2) {
            (Allocation::Heap, _) | (_, Allocation::Heap) => Allocation::Heap,
            (Allocation::Arena, _) | (_, Allocation::Arena) => Allocation::Arena,
            (Allocation::None, Allocation::None) => Allocation::None,
        }
    }

    fn merge_concurrency(&self, c1: Concurrency, c2: Concurrency) -> Concurrency {
        match (c1, c2) {
            (Concurrency::Concurrent, _) | (_, Concurrency::Concurrent) => Concurrency::Concurrent,
            _ => Concurrency::Single,
        }
    }

    /// Infer effects from an expression
    /// Returns the effect set that executing this expression would require
    pub fn infer_expr_effects(&self, expr: &Expr, env_effects: &EffectSet) -> EffectSet {
        match expr {
            // Literals are pure with no allocation
            Expr::IntLit(_, _)
            | Expr::FloatLit(_, _)
            | Expr::BoolLit(_, _)
            | Expr::StringLit(_, _)
            | Expr::CharLit(_, _)
            | Expr::Unit(_) => EffectSet::pure_cpu_none(),

            // Variables inherit environment effects (conservatively)
            Expr::Var(_, _) => EffectSet::pure_cpu_none(),

            // Binary and unary ops are pure
            Expr::BinOp(_, e1, e2, _) => {
                let eff1 = self.infer_expr_effects(e1, env_effects);
                let eff2 = self.infer_expr_effects(e2, env_effects);
                self.merge(&eff1, &eff2)
            }

            Expr::UnOp(_, e, _) => self.infer_expr_effects(e, env_effects),

            // Control flow merges effects
            Expr::If(cond, then_br, else_br, _) => {
                let cond_eff = self.infer_expr_effects(cond, env_effects);
                let then_eff = self.infer_expr_effects(then_br, env_effects);
                let else_eff = self.infer_expr_effects(else_br, env_effects);
                self.merge(&cond_eff, &self.merge(&then_eff, &else_eff))
            }

            Expr::Let(_, val, body, _) => {
                let val_eff = self.infer_expr_effects(val, env_effects);
                let body_eff = self.infer_expr_effects(body, env_effects);
                self.merge(&val_eff, &body_eff)
            }

            Expr::Match(scrutinee, arms, _) => {
                let mut eff = self.infer_expr_effects(scrutinee, env_effects);
                for arm in arms {
                    let arm_eff = self.infer_expr_effects(&arm.body, env_effects);
                    eff = self.merge(&eff, &arm_eff);
                }
                eff
            }

            // Array operations
            Expr::ArrayLit(elems, _) => {
                let mut eff = EffectSet::pure_cpu_none().with_allocation(Allocation::Arena);
                for elem in elems {
                    let elem_eff = self.infer_expr_effects(elem, env_effects);
                    eff = self.merge(&eff, &elem_eff);
                }
                eff
            }

            Expr::ArrayIndex(arr, idx, _) => {
                let arr_eff = self.infer_expr_effects(arr, env_effects);
                let idx_eff = self.infer_expr_effects(idx, env_effects);
                self.merge(&arr_eff, &idx_eff)
            }

            // Parallel primitives require concurrent effect
            Expr::ParFor(_, _, _, _) | Expr::ParMap(_, _, _) | Expr::ParMapInplace(_, _, _, _) => {
                EffectSet::pure_cpu_none().with_concurrency(Concurrency::Concurrent)
            }

            // Async/await require IO
            Expr::Async(_, _) | Expr::Await(_, _) => EffectSet::new()
                .with_purity(Purity::Io)
                .with_concurrency(Concurrency::Concurrent),

            // Actors require IO and concurrent
            Expr::Spawn(_, _) | Expr::Send(_, _, _) | Expr::Receive(_) => EffectSet::new()
                .with_purity(Purity::Io)
                .with_allocation(Allocation::Heap)
                .with_concurrency(Concurrency::Concurrent),

            // Memory operations
            Expr::NewArena(_) => EffectSet::pure_cpu_none().with_allocation(Allocation::Arena),

            Expr::AllocArray(_, _, space, _) => EffectSet::pure_cpu_none()
                .with_allocation(Allocation::Arena)
                .with_execution(match space {
                    Space::Cpu => Execution::Cpu,
                    Space::Gpu => Execution::Gpu,
                }),

            // GPU operations
            Expr::GpuKernel(_, _) => {
                EffectSet::pure_cpu_none().with_execution(Execution::Gpu)
            }

            Expr::CpuToGpu(_, _) | Expr::GpuToCpu(_, _) => {
                EffectSet::new().with_purity(Purity::Io).with_allocation(Allocation::Heap)
            }

            // Debug operations
            Expr::Log(_, _) | Expr::Assert(_, _, _) => {
                EffectSet::new().with_purity(Purity::Debug)
            }

            // Unsafe inherits its content's effects (but should be checked separately)
            Expr::Unsafe(inner, _) => self.infer_expr_effects(inner, env_effects),

            // Function application and others inherit environment effects
            Expr::App(_, _, _) | Expr::Lambda(_, _, _) | Expr::Constructor(_, _, _) => {
                env_effects.clone()
            }

            Expr::Seq(exprs, _) => {
                let mut eff = EffectSet::pure_cpu_none();
                for expr in exprs {
                    let expr_eff = self.infer_expr_effects(expr, env_effects);
                    eff = self.merge(&eff, &expr_eff);
                }
                eff
            }
        }
    }
}

impl Default for EffectChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_purity_checking() {
        let checker = EffectChecker::new();

        let pure = EffectSet::pure_cpu_none();
        let io = EffectSet::new().with_purity(Purity::Io);

        assert!(checker.can_call(&io, &pure).is_ok());
        assert!(checker.can_call(&pure, &io).is_err());
    }

    #[test]
    fn test_allocation_checking() {
        let checker = EffectChecker::new();

        let none = EffectSet::pure_cpu_none();
        let arena = EffectSet::pure_cpu_none().with_allocation(Allocation::Arena);
        let heap = EffectSet::pure_cpu_none().with_allocation(Allocation::Heap);

        assert!(checker.can_call(&heap, &none).is_ok());
        assert!(checker.can_call(&heap, &arena).is_ok());
        assert!(checker.can_call(&arena, &none).is_ok());
        assert!(checker.can_call(&arena, &heap).is_err());
        assert!(checker.can_call(&none, &arena).is_err());
        assert!(checker.can_call(&none, &heap).is_err());
    }

    #[test]
    fn test_concurrency_checking() {
        let checker = EffectChecker::new();

        let single = EffectSet::pure_cpu_none().with_concurrency(Concurrency::Single);
        let concurrent = EffectSet::pure_cpu_none();

        assert!(checker.can_call(&concurrent, &single).is_ok());
        assert!(checker.can_call(&single, &concurrent).is_err());
    }
}
