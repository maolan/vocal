use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RvcConfig {
    pub burnpack_path: Option<PathBuf>,
    pub target_sr: u32,
    pub hop_length: usize,
    pub pitch_shift_semitones: i32,
    pub index_mix: f32,
    pub rms_mix_rate: f32,
    pub formant_shift: f32,
    pub protect: f32,
    pub force_female_extreme: bool,
}

impl Default for RvcConfig {
    fn default() -> Self {
        Self {
            burnpack_path: None,
            target_sr: 48_000,
            hop_length: 160,
            pitch_shift_semitones: 0,
            index_mix: 0.0,
            rms_mix_rate: 0.0,
            formant_shift: 1.0,
            protect: 0.33,
            force_female_extreme: false,
        }
    }
}
