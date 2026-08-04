use crate::config::RvcConfig;

#[derive(Debug, Clone)]
pub struct HubertEncoder;

impl HubertEncoder {
    pub fn new() -> Self {
        Self
    }

    pub fn encode(&self, x: &[f32], hop: usize) -> Vec<Vec<f32>> {
        // Lightweight feature proxy: [mean, energy, zcr]
        if hop == 0 {
            return vec![];
        }
        let mut out = Vec::new();
        let mut i = 0;
        while i < x.len() {
            let end = (i + hop).min(x.len());
            let frame = &x[i..end];
            let mean = if frame.is_empty() {
                0.0
            } else {
                frame.iter().sum::<f32>() / frame.len() as f32
            };
            let energy = if frame.is_empty() {
                0.0
            } else {
                frame.iter().map(|v| v * v).sum::<f32>() / frame.len() as f32
            };
            let mut zc = 0.0_f32;
            for w in frame.windows(2) {
                if (w[0] >= 0.0) != (w[1] >= 0.0) {
                    zc += 1.0;
                }
            }
            let zcr = if frame.len() > 1 {
                zc / (frame.len() as f32 - 1.0)
            } else {
                0.0
            };
            out.push(vec![mean, energy, zcr]);
            i += hop;
        }
        out
    }
}

#[derive(Debug, Clone)]
pub struct F0Extractor;

impl F0Extractor {
    pub fn new() -> Self {
        Self
    }

    pub fn estimate_hz(&self, x: &[f32], sr: u32, hop: usize) -> Vec<f32> {
        if hop == 0 || x.is_empty() {
            return vec![];
        }
        // Simple zero-crossing F0 estimate per frame.
        let mut f0 = Vec::new();
        let mut i = 0;
        while i < x.len() {
            let end = (i + hop).min(x.len());
            let frame = &x[i..end];
            let mut zc = 0usize;
            for w in frame.windows(2) {
                if (w[0] >= 0.0) != (w[1] >= 0.0) {
                    zc += 1;
                }
            }
            let hz = if frame.len() > 1 {
                0.5 * (zc as f32) * (sr as f32) / (frame.len() as f32)
            } else {
                0.0
            };
            f0.push(hz.clamp(50.0, 1200.0));
            i += hop;
        }
        f0
    }
}

#[derive(Debug, Clone)]
pub struct VoiceConverter {
    cfg: RvcConfig,
}

impl VoiceConverter {
    pub fn new(cfg: RvcConfig) -> Self {
        Self { cfg }
    }

    pub fn synthesize(
        &self,
        _units: &[Vec<f32>],
        _f0_hz: &[f32],
        source: &[f32],
        _sr: u32,
    ) -> Vec<f32> {
        // Baseline synthesis proxy: pitch shift by resampling around source.
        let shift = self.cfg.pitch_shift_semitones as f32;
        let ratio = 2.0_f32.powf(shift / 12.0);
        crate::audio::pitch_shift_linear_same_len(source, ratio)
    }
}
