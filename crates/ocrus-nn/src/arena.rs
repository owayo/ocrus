/// Reusable buffer pool for tensor data during inference.
/// Avoids repeated heap allocation across inference runs.
pub struct TensorArena {
    buffers: Vec<Vec<f32>>,
    next: usize,
}

impl TensorArena {
    pub fn new() -> Self {
        Self {
            buffers: Vec::new(),
            next: 0,
        }
    }

    /// Get a buffer of at least `size` elements, zeroed.
    pub fn alloc(&mut self, size: usize) -> &mut Vec<f32> {
        if self.next < self.buffers.len() {
            let buf = &mut self.buffers[self.next];
            buf.clear();
            buf.resize(size, 0.0);
            self.next += 1;
            &mut self.buffers[self.next - 1]
        } else {
            self.buffers.push(vec![0.0; size]);
            self.next += 1;
            self.buffers.last_mut().unwrap()
        }
    }

    /// Reset the arena for the next inference run.
    /// Keeps allocated memory for reuse.
    pub fn reset(&mut self) {
        self.next = 0;
    }
}

impl Default for TensorArena {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc_and_reset() {
        let mut arena = TensorArena::new();

        let buf1 = arena.alloc(100);
        assert_eq!(buf1.len(), 100);
        buf1[0] = 42.0;

        let buf2 = arena.alloc(200);
        assert_eq!(buf2.len(), 200);

        arena.reset();

        // After reset, buffers are reused
        let buf3 = arena.alloc(50);
        assert_eq!(buf3.len(), 50);
        // Data is zeroed on reuse
        assert_eq!(buf3[0], 0.0);
    }

    #[test]
    fn test_growing() {
        let mut arena = TensorArena::new();
        for i in 0..10 {
            arena.alloc(100 * (i + 1));
        }
        assert_eq!(arena.buffers.len(), 10);

        arena.reset();
        // Reuse existing buffers
        for i in 0..10 {
            arena.alloc(50 * (i + 1));
        }
        // No new allocations
        assert_eq!(arena.buffers.len(), 10);
    }
}
