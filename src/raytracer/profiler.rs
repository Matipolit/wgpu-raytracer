//! `RT_PASS_TIMES=1`: GPU time of each pass group per frame (timestamps between the
//! groups, read back synchronously after every submit), averaged and printed every
//! 120 frames. A profiling aid only: the blocking readback serializes CPU and GPU.

pub struct PassProfiler {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    period_ms: f64,
    labels: Vec<&'static str>,
    /// Labels of the marks written this frame (the first is the frame start).
    frame: Vec<&'static str>,
    sums: Vec<f64>,
    frames: u32,
    pending: bool,
}

const SLOTS: u32 = 32;

impl PassProfiler {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        if std::env::var_os("RT_PASS_TIMES").is_none()
            || !device.features().contains(
                wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
            )
        {
            return None;
        }
        let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("pass profiler"),
            ty: wgpu::QueryType::Timestamp,
            count: SLOTS,
        });
        let buffer = |usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pass profiler results"),
                size: u64::from(SLOTS) * 8,
                usage,
                mapped_at_creation: false,
            })
        };
        Some(Self {
            queries,
            resolve: buffer(wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC),
            readback: buffer(wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ),
            period_ms: f64::from(queue.get_timestamp_period()) * 1e-6,
            labels: Vec::new(),
            frame: Vec::new(),
            sums: Vec::new(),
            frames: 0,
            pending: false,
        })
    }

    /// Timestamp the end of the pass group `label` (`"start"` begins the frame).
    pub fn mark(&mut self, encoder: &mut wgpu::CommandEncoder, label: &'static str) {
        if label == "start" {
            self.frame.clear();
        }
        if self.frame.len() as u32 >= SLOTS {
            return;
        }
        encoder.write_timestamp(&self.queries, self.frame.len() as u32);
        self.frame.push(label);
    }

    /// After the frame's last mark: copy the timestamps for readback.
    pub fn finish(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let n = self.frame.len() as u32;
        if n < 2 {
            return;
        }
        encoder.resolve_query_set(&self.queries, 0..n, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.readback, 0, u64::from(n) * 8);
        self.pending = true;
    }

    /// After submit: read this frame's timestamps (blocking) and report periodically.
    pub fn collect(&mut self, device: &wgpu::Device) {
        if !std::mem::take(&mut self.pending) {
            return;
        }
        let slice = self.readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let Ok(view) = slice.get_mapped_range() else {
            return;
        };
        let times: Vec<u64> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        self.readback.unmap();
        if self.labels != self.frame[1..] {
            self.labels = self.frame[1..].to_vec();
            self.sums = vec![0.; self.labels.len()];
            self.frames = 0;
        }
        for (i, sum) in self.sums.iter_mut().enumerate() {
            *sum += times[i + 1].saturating_sub(times[i]) as f64 * self.period_ms;
        }
        self.frames += 1;
        if self.frames == 120 {
            eprintln!("{}", self.report());
            self.sums.iter_mut().for_each(|s| *s = 0.);
            self.frames = 0;
        }
    }

    pub fn report(&self) -> String {
        let n = f64::from(self.frames.max(1));
        let total: f64 = self.sums.iter().sum::<f64>() / n;
        let parts: Vec<String> = self
            .labels
            .iter()
            .zip(&self.sums)
            .filter(|(_, s)| **s / n >= 0.005)
            .map(|(l, s)| format!("{l} {:.2}", s / n))
            .collect();
        format!("GPU ms/frame: {total:.2} = {}", parts.join(" + "))
    }
}
