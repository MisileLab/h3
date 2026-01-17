use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use tokio::runtime::Runtime;
use tokio::task::JoinHandle;

// Arena allocator for region-based memory management
pub struct Arena {
    blocks: Vec<Vec<u8>>,
    current_offset: usize,
    block_size: usize,
}

impl Arena {
    pub fn new() -> Self {
        Arena::with_capacity(1024 * 1024) // 1MB blocks
    }

    pub fn with_capacity(block_size: usize) -> Self {
        Arena {
            blocks: vec![vec![0; block_size]],
            current_offset: 0,
            block_size,
        }
    }

    pub fn allocate(&mut self, size: usize) -> *mut u8 {
        if self.current_offset + size > self.block_size {
            self.blocks.push(vec![0; self.block_size]);
            self.current_offset = 0;
        }

        let ptr = self.blocks.last_mut().unwrap().as_mut_ptr();
        let offset = self.current_offset;
        self.current_offset += size;

        unsafe { ptr.add(offset) }
    }

    pub fn reset(&mut self) {
        self.current_offset = 0;
        for block in &mut self.blocks {
            block.fill(0);
        }
    }
}

impl Default for Arena {
    fn default() -> Self {
        Self::new()
    }
}

// Array type with arena and space tracking
pub struct FluxArray<T> {
    data: Vec<T>,
    space: Space,
    #[allow(dead_code)]
    arena_id: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Space {
    Cpu,
    Gpu,
}

impl<T> FluxArray<T> {
    pub fn new_cpu(data: Vec<T>) -> Self {
        FluxArray {
            data,
            space: Space::Cpu,
            arena_id: None,
        }
    }

    pub fn new_gpu(data: Vec<T>) -> Self {
        FluxArray {
            data,
            space: Space::Gpu,
            arena_id: None,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn space(&self) -> Space {
        self.space
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

// Parallel primitives
pub fn par_for<F>(start: i32, end: i32, f: F)
where
    F: Fn(i32) + Send + Sync,
{
    (start..end).into_par_iter().for_each(f);
}

pub fn par_map<T, U, F>(arr: &[T], f: F) -> Vec<U>
where
    T: Send + Sync,
    U: Send,
    F: Fn(&T) -> U + Send + Sync,
{
    arr.par_iter().map(f).collect()
}

pub fn par_map_inplace<T, U, F>(src: &[T], dst: &mut [U], f: F)
where
    T: Send + Sync,
    U: Send,
    F: Fn(&T) -> U + Send + Sync,
{
    dst.par_iter_mut()
        .zip(src.par_iter())
        .for_each(|(d, s)| *d = f(s));
}

// Task-based async runtime
pub struct Task<T> {
    handle: Arc<Mutex<Option<JoinHandle<T>>>>,
}

impl<T: Send + 'static> Task<T> {
    pub fn spawn<F>(f: F) -> Self
    where
        F: FnOnce() -> T + Send + 'static,
    {
        let runtime = Runtime::new().unwrap();
        let handle = runtime.spawn(async move { f() });

        Task {
            handle: Arc::new(Mutex::new(Some(handle))),
        }
    }

    pub async fn await_result(self) -> Option<T> {
        let handle = self.handle.lock().unwrap().take()?;
        handle.await.ok()
    }

    pub fn block_on(self) -> Option<T> {
        let handle = self.handle.lock().unwrap().take()?;
        Runtime::new().unwrap().block_on(handle).ok()
    }
}

// GPU primitives (simplified stubs)
pub struct GpuKernel<T> {
    data: Vec<T>,
}

impl<T: Clone> GpuKernel<T> {
    pub fn new(data: Vec<T>) -> Self {
        GpuKernel { data }
    }

    pub fn execute<F>(&mut self, f: F)
    where
        F: Fn(&T) -> T,
    {
        // In a real implementation, this would compile to GPU code
        // For now, we run on CPU as a fallback
        for item in &mut self.data {
            *item = f(item);
        }
    }

}

pub fn cpu_to_gpu<T: Clone>(data: Vec<T>) -> GpuKernel<T> {
    GpuKernel::new(data)
}

pub fn gpu_to_cpu<T>(kernel: GpuKernel<T>) -> Vec<T> {
    kernel.data
}

// Debug utilities
pub fn runtime_log<T: std::fmt::Debug>(value: T) {
    println!("[FLUX LOG] {:?}", value);
}

pub fn runtime_assert(condition: bool, message: &str) {
    assert!(condition, "{}", message);
}

// Runtime functions callable from generated code
pub fn runtime_par_for_fn(start: i32, end: i32, _body_fn: usize) {
    // In a full implementation, body_fn would be a function pointer
    // For now, this is a stub
    println!("par_for called: {} to {}", start, end);
}

pub fn runtime_async_fn() -> usize {
    // Return task handle
    0
}

pub fn runtime_await_fn(_task: usize) -> i32 {
    // Await task and return result
    0
}

pub fn runtime_new_arena_fn() -> *mut Arena {
    Box::into_raw(Box::new(Arena::new()))
}

pub fn runtime_free_arena_fn(arena: *mut Arena) {
    if !arena.is_null() {
        unsafe {
            let _ = Box::from_raw(arena);
        }
    }
}

// Memory pool for lock-free allocation
pub struct MemoryPool {
    pools: Vec<Mutex<Vec<Vec<u8>>>>,
}

impl MemoryPool {
    pub fn new(num_threads: usize) -> Self {
        let mut pools = Vec::new();
        for _ in 0..num_threads {
            pools.push(Mutex::new(Vec::new()));
        }

        MemoryPool { pools }
    }

    pub fn allocate(&self, size: usize, thread_id: usize) -> Vec<u8> {
        let pool_idx = thread_id % self.pools.len();
        let mut pool = self.pools[pool_idx].lock().unwrap();

        if let Some(buf) = pool.pop() {
            if buf.len() >= size {
                return buf;
            }
        }

        vec![0; size]
    }

    pub fn deallocate(&self, mut buf: Vec<u8>, thread_id: usize) {
        let pool_idx = thread_id % self.pools.len();
        let mut pool = self.pools[pool_idx].lock().unwrap();

        buf.clear();
        pool.push(buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_allocation() {
        let mut arena = Arena::new();
        let ptr1 = arena.allocate(100);
        let ptr2 = arena.allocate(200);

        assert!(!ptr1.is_null());
        assert!(!ptr2.is_null());
        assert_ne!(ptr1, ptr2);
    }

    #[test]
    fn test_par_for() {
        use std::sync::atomic::{AtomicI32, Ordering};

        let counter = AtomicI32::new(0);
        par_for(0, 10, |_| {
            counter.fetch_add(1, Ordering::SeqCst);
        });

        assert_eq!(counter.load(Ordering::SeqCst), 10);
    }

    #[test]
    fn test_par_map() {
        let data = vec![1, 2, 3, 4, 5];
        let result = par_map(&data, |x| x * 2);

        assert_eq!(result, vec![2, 4, 6, 8, 10]);
    }
}
