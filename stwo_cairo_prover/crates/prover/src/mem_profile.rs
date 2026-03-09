/// Host memory profiling via /proc/self/status.
/// Reports VmRSS (current resident) and VmHWM (peak resident) in MB.

pub struct MemSnapshot {
    pub label: String,
    pub rss_mb: f64,
    pub hwm_mb: f64,
    pub timestamp_ms: u128,
}

impl std::fmt::Display for MemSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[MEM {:>8}ms] {:<45} RSS {:>8.1} MB  HWM {:>8.1} MB",
            self.timestamp_ms, self.label, self.rss_mb, self.hwm_mb
        )
    }
}

/// Read VmRSS and VmHWM from /proc/self/status (Linux only).
fn read_proc_mem() -> (f64, f64) {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let mut rss_kb = 0u64;
    let mut hwm_kb = 0u64;
    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            rss_kb = line
                .split_whitespace()
                .nth(1)
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
        } else if line.starts_with("VmHWM:") {
            hwm_kb = line
                .split_whitespace()
                .nth(1)
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
        }
    }
    (rss_kb as f64 / 1024.0, hwm_kb as f64 / 1024.0)
}

pub struct MemProfiler {
    start: std::time::Instant,
    snapshots: Vec<MemSnapshot>,
}

impl MemProfiler {
    pub fn new() -> Self {
        Self {
            start: std::time::Instant::now(),
            snapshots: Vec::new(),
        }
    }

    /// Take a snapshot with the given label, print it immediately, and store it.
    pub fn snap(&mut self, label: &str) {
        let (rss_mb, hwm_mb) = read_proc_mem();
        let s = MemSnapshot {
            label: label.to_string(),
            rss_mb,
            hwm_mb,
            timestamp_ms: self.start.elapsed().as_millis(),
        };
        eprintln!("{}", s);
        self.snapshots.push(s);
    }

    /// Print a delta report: for each consecutive pair, show RSS change.
    pub fn print_delta_report(&self) {
        eprintln!("\n========== MEMORY DELTA REPORT ==========");
        eprintln!(
            "{:<45} {:>10} {:>10} {:>10}",
            "Phase", "RSS (MB)", "Delta (MB)", "Time (ms)"
        );
        eprintln!("{}", "-".repeat(80));
        for i in 0..self.snapshots.len() {
            let s = &self.snapshots[i];
            let delta = if i > 0 {
                s.rss_mb - self.snapshots[i - 1].rss_mb
            } else {
                0.0
            };
            let dt = if i > 0 {
                s.timestamp_ms - self.snapshots[i - 1].timestamp_ms
            } else {
                0
            };
            eprintln!(
                "{:<45} {:>10.1} {:>+10.1} {:>10}",
                s.label, s.rss_mb, delta, dt
            );
        }
        eprintln!("{}", "-".repeat(80));
        if let (Some(first), Some(last)) = (self.snapshots.first(), self.snapshots.last()) {
            eprintln!(
                "Total: RSS grew {:.1} MB ({:.1} GB), HWM = {:.1} MB ({:.1} GB), elapsed = {}ms",
                last.rss_mb - first.rss_mb,
                (last.rss_mb - first.rss_mb) / 1024.0,
                last.hwm_mb,
                last.hwm_mb / 1024.0,
                last.timestamp_ms,
            );
        }
    }
}
