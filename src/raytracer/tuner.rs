//! Runtime selection between trace-pass variants whose speed depends on the scene
//! and the view (ray-cone texture filtering, paths starting from the visibility
//! buffer). Every frame's trace (+ raster) GPU time is measured with timestamps and
//! read back asynchronously; the tuner tries each variant briefly, keeps the
//! fastest, and re-checks the others periodically or when the frame time drifts
//! (the camera moved somewhere the choice may differ).
#![allow(clippy::cast_possible_truncation)]
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Texture mip selection from ray cones.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cones {
    /// Base level everywhere.
    Off,
    /// Mips from the pixel footprint at the primary hit only.
    Primary,
    /// Cones tracked along the whole path.
    All,
}

/// One trace-pass configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Variant {
    pub cones: Cones,
    pub visibility_primary: bool,
}

/// Frames per variant when exploring (the first one after a switch is discarded).
const EXPLORE_FRAMES: usize = 4;
/// Frames between re-checks of the other variants; doubles while the choice holds.
const RECHECK_MIN: u32 = 120;
const RECHECK_MAX: u32 = 960;
/// Re-check at once when the chosen variant gets this much slower than when chosen.
const DRIFT: f64 = 1.25;
const HISTORY: usize = 4;
/// A variant must be this much faster than the current one to replace it.
const SWITCH_GAIN: f64 = 0.97;

pub struct AutoTuner {
    pub variants: Vec<Variant>,
    current: usize,
    /// Variants to run next (exploration), front first.
    schedule: VecDeque<usize>,
    samples: Vec<VecDeque<f64>>,
    since_check: u32,
    recheck: u32,
    chosen_ms: f64,
    last_variant: Option<usize>,
}

impl AutoTuner {
    pub fn new(variants: Vec<Variant>) -> Self {
        let n = variants.len();
        let mut tuner = Self {
            variants,
            current: 0,
            schedule: VecDeque::new(),
            samples: vec![VecDeque::new(); n],
            since_check: 0,
            recheck: RECHECK_MIN,
            chosen_ms: f64::INFINITY,
            last_variant: None,
        };
        tuner.explore(true);
        tuner
    }

    fn explore(&mut self, all: bool) {
        if self.variants.len() < 2 {
            return;
        }
        for v in 0..self.variants.len() {
            if all || v != self.current {
                self.samples[v].clear();
                self.schedule.extend(std::iter::repeat_n(v, EXPLORE_FRAMES));
            }
        }
        // End on the current variant so its numbers are fresh too.
        self.schedule.extend(std::iter::repeat_n(self.current, 2));
        self.since_check = 0;
    }

    /// Variant to trace this frame.
    pub fn next(&mut self) -> usize {
        self.schedule.pop_front().unwrap_or_else(|| {
            self.since_check += 1;
            if self.since_check >= self.recheck {
                self.explore(false);
                self.recheck = (self.recheck * 2).min(RECHECK_MAX);
                return self.schedule.pop_front().unwrap_or(self.current);
            }
            self.current
        })
    }

    /// A measured frame: `ms` of trace (+ raster) time with variant `v`. `switched`
    /// marks the first frame after a variant change (caches cold), which is dropped.
    pub fn record(&mut self, v: usize, ms: f64, switched: bool) {
        if switched || !ms.is_finite() {
            return;
        }
        let history = &mut self.samples[v];
        history.push_back(ms);
        if history.len() > HISTORY {
            history.pop_front();
        }
        if !self.schedule.is_empty() {
            return;
        }
        let medians: Vec<Option<f64>> = self.samples.iter().map(median).collect();
        let best = medians
            .iter()
            .enumerate()
            .filter_map(|(i, m)| m.map(|m| (i, m)))
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((best, ms)) = best {
            // Hysteresis: near-ties do not flip the choice back and forth.
            let current = medians[self.current].unwrap_or(f64::INFINITY);
            if best != self.current && ms < current * SWITCH_GAIN {
                if crate::has_flag("--tuner-log") {
                    eprintln!(
                        "auto-tune: {:?} {ms:.2} ms, was {:?}",
                        self.variants[best], self.variants[self.current]
                    );
                }
                self.current = best;
                self.chosen_ms = ms;
                self.recheck = RECHECK_MIN;
            }
        }
        if v == self.current {
            if let Some(m) = medians[v] {
                if m < self.chosen_ms {
                    self.chosen_ms = m;
                } else if m > self.chosen_ms * DRIFT && self.samples[v].len() >= HISTORY {
                    if crate::has_flag("--tuner-log") {
                        eprintln!(
                            "auto-tune: {:?} drifted {:.2} -> {m:.2} ms, re-checking",
                            self.variants[v], self.chosen_ms
                        );
                    }
                    self.chosen_ms = m;
                    self.recheck = RECHECK_MIN;
                    self.explore(false);
                }
            }
        }
    }

    pub fn current(&self) -> Variant {
        self.variants[self.current]
    }

    /// Whether `v` differs from the previous frame's variant (and remember it).
    pub fn switched(&mut self, v: usize) -> bool {
        let switched = self.last_variant != Some(v);
        self.last_variant = Some(v);
        switched
    }
}

fn median(s: &VecDeque<f64>) -> Option<f64> {
    if s.len() < 2 {
        return None;
    }
    let mut v: Vec<f64> = s.iter().copied().collect();
    v.sort_by(f64::total_cmp);
    Some(v[v.len() / 2])
}

/// Ring of timestamp query slots read back without stalling the frame.
pub struct FrameTimer {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    slots: Vec<Slot>,
    period_ms: f64,
}

struct Slot {
    readback: wgpu::Buffer,
    state: SlotState,
    mapped: Arc<AtomicBool>,
}

#[derive(Clone, Copy)]
enum SlotState {
    Free,
    /// Encoded this frame; map after the frame is submitted.
    Encoded {
        variant: usize,
        switched: bool,
    },
    Mapping {
        variant: usize,
        switched: bool,
    },
}

const SLOTS: usize = 8;
/// Query resolves must start at multiples of 256 bytes.
const STRIDE: u64 = wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT;

impl FrameTimer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        if !device.features().contains(
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
        ) {
            return None;
        }
        let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("auto-tune timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 2 * SLOTS as u32,
        });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("auto-tune resolve"),
            size: STRIDE * SLOTS as u64,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let slots = (0..SLOTS)
            .map(|_| Slot {
                readback: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("auto-tune readback"),
                    size: 16,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                state: SlotState::Free,
                mapped: Arc::new(AtomicBool::new(false)),
            })
            .collect();
        Some(Self {
            queries,
            resolve,
            slots,
            period_ms: f64::from(queue.get_timestamp_period()) / 1e6,
        })
    }

    /// Finished measurements as (variant, ms, switched).
    pub fn collect(&mut self) -> Vec<(usize, f64, bool)> {
        let mut done = Vec::new();
        for slot in &mut self.slots {
            if let SlotState::Mapping { variant, switched } = slot.state {
                if slot.mapped.swap(false, Ordering::Acquire) {
                    let ms = {
                        let data = slot.readback.slice(..).get_mapped_range().unwrap();
                        let ticks: &[u64] = bytemuck::cast_slice(&data);
                        ticks[1].wrapping_sub(ticks[0]) as f64 * self.period_ms
                    };
                    slot.readback.unmap();
                    slot.state = SlotState::Free;
                    done.push((variant, ms, switched));
                }
            }
        }
        done
    }

    /// Start timing; returns the slot, or None when all slots are in flight.
    pub fn begin(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        variant: usize,
        switched: bool,
    ) -> Option<usize> {
        let index = self
            .slots
            .iter()
            .position(|s| matches!(s.state, SlotState::Free))?;
        self.slots[index].state = SlotState::Encoded { variant, switched };
        encoder.write_timestamp(&self.queries, 2 * index as u32);
        Some(index)
    }

    pub fn end(&self, encoder: &mut wgpu::CommandEncoder, index: usize) {
        let first = 2 * index as u32;
        encoder.write_timestamp(&self.queries, first + 1);
        encoder.resolve_query_set(
            &self.queries,
            first..first + 2,
            &self.resolve,
            STRIDE * index as u64,
        );
        encoder.copy_buffer_to_buffer(
            &self.resolve,
            STRIDE * index as u64,
            &self.slots[index].readback,
            0,
            16,
        );
    }

    /// Call after submitting the frame's commands.
    pub fn after_submit(&mut self) {
        for slot in &mut self.slots {
            if let SlotState::Encoded { variant, switched } = slot.state {
                slot.state = SlotState::Mapping { variant, switched };
                let mapped = Arc::clone(&slot.mapped);
                slot.readback
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |r| {
                        if r.is_ok() {
                            mapped.store(true, Ordering::Release);
                        }
                    });
            }
        }
    }
}

/// Traversal stack of the 4-wide BVH (startup level: changes the shader source).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stack {
    /// Workgroup (shared) memory, 32 entries per invocation, exact; 64-lane workgroups.
    Shared32,
    /// Registers, 16 entries; a full stack drops entries (rare speckles).
    Registers16,
    /// Module-scope private array, 64 entries, exact.
    Private64,
}

/// Startup-level configuration of the trace kernel's module and pipelines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModuleConfig {
    /// None: not the 4-wide BVH (or a stack experiment set through RT_* variables).
    pub stack: Option<Stack>,
    pub workgroup: [u32; 2],
}

impl ModuleConfig {
    fn encode(&self) -> String {
        let stack = match self.stack {
            None => "none",
            Some(Stack::Shared32) => "shared32",
            Some(Stack::Registers16) => "registers16",
            Some(Stack::Private64) => "private64",
        };
        format!("{stack} {}x{}", self.workgroup[0], self.workgroup[1])
    }

    fn decode(text: &str) -> Option<Self> {
        let (stack, size) = text.trim().split_once(' ')?;
        let stack = match stack {
            "none" => None,
            "shared32" => Some(Stack::Shared32),
            "registers16" => Some(Stack::Registers16),
            "private64" => Some(Stack::Private64),
            _ => return None,
        };
        let (x, y) = size.split_once('x')?;
        Some(Self {
            stack,
            workgroup: [x.parse().ok()?, y.parse().ok()?],
        })
    }
}

/// Calibration results per GPU, driver and scene: `$XDG_CACHE_HOME/wgpu-raytracer/tuning.txt`
/// (or ~/.cache/...), one `key<TAB>config` line each.
fn cache_path() -> Option<std::path::PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))?;
    Some(base.join("wgpu-raytracer").join("tuning.txt"))
}

pub fn load_cached(key: &str) -> Option<ModuleConfig> {
    let text = std::fs::read_to_string(cache_path()?).ok()?;
    text.lines()
        .filter_map(|line| line.split_once('\t'))
        .filter(|(k, _)| *k == key)
        .last()
        .and_then(|(_, v)| ModuleConfig::decode(v))
}

pub fn store_cached(key: &str, config: ModuleConfig) {
    let Some(path) = cache_path() else { return };
    let _ = std::fs::create_dir_all(path.parent().unwrap());
    let mut text = std::fs::read_to_string(&path).unwrap_or_default();
    text = text
        .lines()
        .filter(|l| !l.starts_with(&format!("{key}\t")))
        .map(|l| format!("{l}\n"))
        .collect();
    text.push_str(&format!("{key}\t{}\n", config.encode()));
    let _ = std::fs::write(path, text);
}
