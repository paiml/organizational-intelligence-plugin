//! Performance Tuning Utilities
//!
//! PROD-005: Optimizations for production workloads
//! Provides batching, caching, and parallel processing utilities

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Batch processor for efficient bulk operations
pub struct BatchProcessor<T> {
    batch_size: usize,
    buffer: Vec<T>,
}

impl<T> BatchProcessor<T> {
    /// Create new batch processor
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            buffer: Vec::with_capacity(batch_size),
        }
    }

    /// Add item to batch, returns batch if full
    pub fn add(&mut self, item: T) -> Option<Vec<T>> {
        self.buffer.push(item);
        if self.buffer.len() >= self.batch_size {
            Some(self.flush())
        } else {
            None
        }
    }

    /// Flush remaining items
    pub fn flush(&mut self) -> Vec<T> {
        std::mem::take(&mut self.buffer)
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get current buffer size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}

/// Simple LRU cache for expensive computations
pub struct LruCache<K, V> {
    capacity: usize,
    map: HashMap<K, (V, Instant)>,
    ttl: Option<Duration>,
}

impl<K: Eq + Hash + Clone, V: Clone> LruCache<K, V> {
    /// Create cache with capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            map: HashMap::with_capacity(capacity),
            ttl: None,
        }
    }

    /// Create cache with TTL
    pub fn with_ttl(capacity: usize, ttl: Duration) -> Self {
        Self {
            capacity,
            map: HashMap::with_capacity(capacity),
            ttl: Some(ttl),
        }
    }

    /// Get value from cache
    pub fn get(&self, key: &K) -> Option<V> {
        self.map.get(key).and_then(|(v, inserted)| {
            if let Some(ttl) = self.ttl {
                if inserted.elapsed() > ttl {
                    return None;
                }
            }
            Some(v.clone())
        })
    }

    /// Insert value into cache
    pub fn insert(&mut self, key: K, value: V) {
        // Evict oldest if at capacity
        if self.map.len() >= self.capacity && !self.map.contains_key(&key) {
            if let Some(oldest_key) = self.find_oldest() {
                self.map.remove(&oldest_key);
            }
        }
        self.map.insert(key, (value, Instant::now()));
    }

    /// Check if key exists
    pub fn contains(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Clear cache
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Get cache size
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    fn find_oldest(&self) -> Option<K> {
        self.map
            .iter()
            .min_by_key(|(_, (_, instant))| *instant)
            .map(|(k, _)| k.clone())
    }
}

/// Memory-efficient ring buffer for streaming data
pub struct RingBuffer<T> {
    buffer: Vec<Option<T>>,
    head: usize,
    tail: usize,
    size: usize,
}

impl<T: Clone> RingBuffer<T> {
    /// Create ring buffer with capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![None; capacity],
            head: 0,
            tail: 0,
            size: 0,
        }
    }

    /// Push item (overwrites oldest if full)
    pub fn push(&mut self, item: T) {
        self.buffer[self.tail] = Some(item);
        self.tail = (self.tail + 1) % self.buffer.len();
        if self.size < self.buffer.len() {
            self.size += 1;
        } else {
            self.head = (self.head + 1) % self.buffer.len();
        }
    }

    /// Pop oldest item
    pub fn pop(&mut self) -> Option<T> {
        if self.size == 0 {
            return None;
        }
        let item = self.buffer[self.head].take();
        self.head = (self.head + 1) % self.buffer.len();
        self.size -= 1;
        item
    }

    /// Get current size
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Check if full
    pub fn is_full(&self) -> bool {
        self.size == self.buffer.len()
    }

    /// Get all items as vector
    pub fn to_vec(&self) -> Vec<T> {
        let mut result = Vec::with_capacity(self.size);
        let mut idx = self.head;
        for _ in 0..self.size {
            if let Some(ref item) = self.buffer[idx] {
                result.push(item.clone());
            }
            idx = (idx + 1) % self.buffer.len();
        }
        result
    }
}

/// Performance statistics collector
#[derive(Debug, Clone, Default)]
pub struct PerfStats {
    pub operation_count: u64,
    pub total_duration_ns: u64,
    pub min_duration_ns: u64,
    pub max_duration_ns: u64,
}

impl PerfStats {
    pub fn new() -> Self {
        Self {
            operation_count: 0,
            total_duration_ns: 0,
            min_duration_ns: u64::MAX,
            max_duration_ns: 0,
        }
    }

    /// Record an operation duration
    pub fn record(&mut self, duration_ns: u64) {
        self.operation_count += 1;
        self.total_duration_ns += duration_ns;
        self.min_duration_ns = self.min_duration_ns.min(duration_ns);
        self.max_duration_ns = self.max_duration_ns.max(duration_ns);
    }

    /// Get average duration in nanoseconds
    pub fn avg_ns(&self) -> u64 {
        if self.operation_count == 0 {
            0
        } else {
            self.total_duration_ns / self.operation_count
        }
    }

    /// Get average duration in microseconds
    pub fn avg_us(&self) -> f64 {
        self.avg_ns() as f64 / 1000.0
    }

    /// Get average duration in milliseconds
    pub fn avg_ms(&self) -> f64 {
        self.avg_ns() as f64 / 1_000_000.0
    }

    /// Get throughput (operations per second)
    pub fn throughput(&self) -> f64 {
        if self.total_duration_ns == 0 {
            0.0
        } else {
            self.operation_count as f64 / (self.total_duration_ns as f64 / 1_000_000_000.0)
        }
    }
}

/// Scoped timer for measuring code blocks
pub struct ScopedTimer<'a> {
    stats: &'a mut PerfStats,
    start: Instant,
}

impl<'a> ScopedTimer<'a> {
    pub fn new(stats: &'a mut PerfStats) -> Self {
        Self {
            stats,
            start: Instant::now(),
        }
    }
}

impl<'a> Drop for ScopedTimer<'a> {
    fn drop(&mut self) {
        let duration = self.start.elapsed().as_nanos() as u64;
        self.stats.record(duration);
    }
}

/// Parallel chunk processor
pub fn process_chunks<T, R, F>(items: Vec<T>, chunk_size: usize, f: F) -> Vec<R>
where
    T: Send + Sync,
    R: Send,
    F: Fn(&[T]) -> Vec<R> + Send + Sync,
{
    let chunks: Vec<_> = items.chunks(chunk_size).collect();
    let f = Arc::new(f);

    // Process sequentially (parallel version would use rayon)
    chunks.iter().flat_map(|chunk| f(chunk)).collect()
}

/// Estimate memory usage for feature vectors
pub fn estimate_memory_bytes(feature_count: usize, dimensions: usize) -> usize {
    // Each feature: dimensions * sizeof(f32) + struct overhead
    let feature_size = dimensions * std::mem::size_of::<f32>() + 64; // 64 bytes struct overhead
    feature_count * feature_size
}

/// Format bytes as human-readable string
pub fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_processor() {
        let mut processor = BatchProcessor::new(3);

        assert!(processor.add(1).is_none());
        assert!(processor.add(2).is_none());

        let batch = processor.add(3);
        assert!(batch.is_some());
        assert_eq!(batch.unwrap(), vec![1, 2, 3]);

        processor.add(4);
        let remaining = processor.flush();
        assert_eq!(remaining, vec![4]);
    }

    #[test]
    fn test_lru_cache() {
        let mut cache = LruCache::new(2);

        cache.insert("a", 1);
        cache.insert("b", 2);

        assert_eq!(cache.get(&"a"), Some(1));
        assert_eq!(cache.get(&"b"), Some(2));

        // This should evict "a" (oldest)
        cache.insert("c", 3);
        assert_eq!(cache.get(&"a"), None);
        assert_eq!(cache.get(&"c"), Some(3));
    }

    #[test]
    fn test_ring_buffer() {
        let mut buffer = RingBuffer::new(3);

        buffer.push(1);
        buffer.push(2);
        buffer.push(3);
        assert!(buffer.is_full());

        // Overwrites oldest (1)
        buffer.push(4);
        assert_eq!(buffer.to_vec(), vec![2, 3, 4]);

        assert_eq!(buffer.pop(), Some(2));
        assert_eq!(buffer.len(), 2);
    }

    #[test]
    fn test_perf_stats() {
        let mut stats = PerfStats::new();

        stats.record(1000);
        stats.record(2000);
        stats.record(3000);

        assert_eq!(stats.operation_count, 3);
        assert_eq!(stats.avg_ns(), 2000);
        assert_eq!(stats.min_duration_ns, 1000);
        assert_eq!(stats.max_duration_ns, 3000);
    }

    #[test]
    fn test_scoped_timer() {
        let mut stats = PerfStats::new();

        {
            let _timer = ScopedTimer::new(&mut stats);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        assert_eq!(stats.operation_count, 1);
        assert!(stats.avg_ns() > 0);
    }

    #[test]
    fn test_estimate_memory() {
        let bytes = estimate_memory_bytes(1000, 8);
        assert!(bytes > 0);
        // 1000 features * (8 * 4 bytes + 64 overhead) = 96,000 bytes
        assert!(bytes >= 96_000);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 bytes");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }

    #[test]
    fn test_process_chunks() {
        let items = vec![1, 2, 3, 4, 5];
        let results = process_chunks(items, 2, |chunk| chunk.iter().map(|x| x * 2).collect());
        assert_eq!(results, vec![2, 4, 6, 8, 10]);
    }
}
