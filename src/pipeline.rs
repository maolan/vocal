use anyhow::Result;

use crate::audio;
use crate::burn_rvc::BurnRvcModel;
use crate::config::RvcConfig;
use crate::retrieval::FeatureIndex;
use crate::rvc::{F0Extractor, HubertEncoder, VoiceConverter};

pub struct RvcPipeline {
    cfg: RvcConfig,
    hubert: HubertEncoder,
    f0: F0Extractor,
    index: FeatureIndex,
    vc: VoiceConverter,
    burn_model: Option<BurnRvcModel>,
}

impl RvcPipeline {
    pub fn new(cfg: RvcConfig) -> Result<Self> {
        let vc = VoiceConverter::new(cfg.clone());
        let burn_model = match &cfg.burnpack_path {
            Some(path) => Some(BurnRvcModel::from_burnpack(path)?),
            None => None,
        };
        Ok(Self {
            cfg,
            hubert: HubertEncoder::new(),
            f0: F0Extractor::new(),
            index: FeatureIndex::empty(),
            vc,
            burn_model,
        })
    }

    pub fn target_sr(&self) -> u32 {
        self.cfg.target_sr
    }

    pub fn infer(&self, x: &[f32], sr: u32) -> Result<Vec<f32>> {
        let x48 = audio::resample_linear(x, sr, self.cfg.target_sr);

        let units = self.hubert.encode(&x48, self.cfg.hop_length);
        let units = self.index.blend(&units, self.cfg.index_mix);

        let mut f0 = self
            .f0
            .estimate_hz(&x48, self.cfg.target_sr, self.cfg.hop_length);
        let shift_ratio = 2.0_f32.powf(self.cfg.pitch_shift_semitones as f32 / 12.0);
        crate::simd::mul_inplace(&mut f0, shift_ratio);

        let mut y = self.vc.synthesize(&units, &f0, &x48, self.cfg.target_sr);

        if self.cfg.rms_mix_rate > 0.0 {
            let in_r = audio::rms(&x48);
            let out_r = audio::rms(&y);
            if out_r > 1.0e-7 {
                let gain = (in_r / out_r).powf(self.cfg.rms_mix_rate.clamp(0.0, 1.0));
                crate::simd::mul_inplace(&mut y, gain);
            }
        }
        if let Some(model) = &self.burn_model {
            apply_burnpack_voice_coloration(
                &mut y,
                &f0,
                self.cfg.hop_length,
                self.cfg.target_sr,
                model.profile,
                model.neural.as_ref(),
                self.cfg.force_female_extreme,
            );
            // Model takeover: when no explicit pitch shift is requested, force
            // a clear identity move so output is unmistakably different.
            if self.cfg.pitch_shift_semitones == 0 {
                let auto_semitones = if self.cfg.force_female_extreme { 12 } else { 7 };
                let ratio = 2.0_f32.powf(auto_semitones as f32 / 12.0);
                y = audio::pitch_shift_linear_same_len(&y, ratio);
            }

            // Strong post formant/brightness tilt for female-target character.
            let mut lp = 0.0_f32;
            let mut hp_prev_x = 0.0_f32;
            let mut hp_prev_y = 0.0_f32;
            let lp_alpha = if self.cfg.force_female_extreme {
                0.30
            } else {
                0.22
            };
            let hp_alpha = if self.cfg.force_female_extreme {
                0.975
            } else {
                0.95
            };
            for s in &mut y {
                let x = *s;
                lp += lp_alpha * (x - lp);
                let hp = hp_alpha * (hp_prev_y + x - hp_prev_x);
                hp_prev_x = x;
                hp_prev_y = hp;
                let shaped = if self.cfg.force_female_extreme {
                    (lp * 0.35 + hp * 1.05).tanh()
                } else {
                    (lp * 0.55 + hp * 0.75).tanh()
                };
                *s = shaped.clamp(-0.98, 0.98);
            }
        }
        Ok(y)
    }
}

fn apply_burnpack_voice_coloration(
    y: &mut [f32],
    f0: &[f32],
    hop: usize,
    sr: u32,
    profile: crate::burn_rvc::RvcProfile,
    neural: Option<&crate::burn_rvc::NeuralCore>,
    force_female_extreme: bool,
) {
    if y.is_empty() {
        return;
    }
    let source = y.to_vec();
    let feminine_shift = 1.0 + profile.femininity * 0.35 * profile.speaker_strength;
    let frame_count = if hop == 0 {
        0
    } else {
        source.len().div_ceil(hop)
    };
    let frame_styles: Option<Vec<StyleFrame>> = neural.map(|core| {
        let mut styles = Vec::with_capacity(frame_count);
        for frame in 0..frame_count {
            let center = frame * hop;
            let unit = unit_proxy(center, hop, &source);
            let hz = f0
                .get(frame.min(f0.len().saturating_sub(1)))
                .copied()
                .unwrap_or(160.0)
                .clamp(50.0, 1200.0)
                * feminine_shift;
            styles.push(neural_style_forward(core, unit, hz));
        }
        styles
    });

    let mut lp = 0.0_f32;
    let mut hp_prev_x = 0.0_f32;
    let mut hp_prev_y = 0.0_f32;
    let mut fir_hist = [0.0_f32; 7];
    let nyq = (sr as f32 * 0.5).max(1.0);
    let formant_hz = (900.0 * profile.formant_shift).clamp(200.0, 4_000.0);
    let lp_alpha = (formant_hz / nyq).clamp(0.01, 0.95);
    let hp_alpha = (0.92 + profile.brightness * 0.06).clamp(0.90, 0.995);
    let drive = 1.0 + profile.drive * 3.0;
    let identity_mix = if force_female_extreme {
        1.0
    } else {
        (0.82 + 0.16 * profile.speaker_strength).clamp(0.80, 0.99)
    };
    let mut tilt_lp = 0.0_f32;
    let mut band_lp = [0.0_f32; 8];
    let mut up_hists: Vec<Vec<Vec<f32>>> = neural
        .map(|n| {
            n.up_kernels
                .iter()
                .map(|stage| {
                    stage
                        .kernels
                        .iter()
                        .map(|k| vec![0.0_f32; k.len()])
                        .collect()
                })
                .collect()
        })
        .unwrap_or_default();
    let mut res_hists: Vec<Vec<Vec<f32>>> = neural
        .map(|n| {
            n.res_stages
                .iter()
                .map(|stage| {
                    stage
                        .kernels
                        .iter()
                        .map(|k| vec![0.0_f32; k.len()])
                        .collect()
                })
                .collect()
        })
        .unwrap_or_default();

    for (i, s) in y.iter_mut().enumerate() {
        let mut neural_tilt = 0.0_f32;
        let mut neural_air = 0.0_f32;
        let mut neural_formant = 1.0_f32;
        let mut band_gains = [0.0_f32; 8];
        if let Some(styles) = &frame_styles {
            let frame = i.checked_div(hop).unwrap_or(0);
            let style = styles
                .get(frame.min(styles.len().saturating_sub(1)))
                .copied()
                .unwrap_or_default();
            neural_tilt = style.tilt;
            neural_air = style.air;
            neural_formant = style.formant;
            band_gains = style.band_gains;
        }

        let x = *s;
        lp = lp + lp_alpha * (x - lp);
        let hp = hp_alpha * (hp_prev_y + x - hp_prev_x);
        hp_prev_x = x;
        hp_prev_y = hp;
        let airy =
            hp * (0.08 + 0.08 * profile.femininity * profile.speaker_strength + neural_air * 0.08);
        let formant_mix = (0.75 + profile.brightness * 0.25) * neural_formant.clamp(0.7, 1.4);
        // Source-filter style transform: preserve content while shifting timbre.
        tilt_lp += 0.03 * (x - tilt_lp);
        let tilt_hp = x - tilt_lp;
        let tilt = (0.75 + neural_tilt * 0.35).clamp(0.45, 1.35);
        // Optional checkpoint-derived decoder FIR shaping.
        let mut decoder_shaped = x;
        if let Some(core) = neural {
            fir_hist.rotate_right(1);
            fir_hist[0] = x;
            let mut pre = 0.0_f32;
            let mut post = 0.0_f32;
            for (k, v) in fir_hist.iter().enumerate() {
                pre += *v * core.conv_pre_kernel[k];
                post += *v * core.conv_post_kernel[k];
            }
            decoder_shaped = (pre * 0.6 + post * 0.4).tanh();

            // Decoder upsample-stack inspired shaping (simplified).
            if !core.up_kernels.is_empty() && up_hists.len() == core.up_kernels.len() {
                let mut stage_in = x;
                for (stage_idx, stage) in core.up_kernels.iter().enumerate() {
                    let stage_hists = &mut up_hists[stage_idx];
                    let mut accum = 0.0_f32;
                    let mut n = 0usize;
                    for (g, kernel) in stage.kernels.iter().enumerate() {
                        let hist = &mut stage_hists[g];
                        if hist.is_empty() {
                            continue;
                        }
                        hist.rotate_right(1);
                        hist[0] = stage_in;
                        let mut yk = 0.0_f32;
                        for i in 0..kernel.len() {
                            yk += hist[i] * kernel[i];
                        }
                        accum += yk;
                        n += 1;
                    }
                    if n > 0 {
                        stage_in = (accum / n as f32).tanh();
                    }
                }
                decoder_shaped = (decoder_shaped * 0.6 + stage_in * 0.4).tanh();
            }

            // Lightweight residual stack approximation from dec.resblocks.
            if !core.res_stages.is_empty() && res_hists.len() == core.res_stages.len() {
                let mut rs = decoder_shaped;
                for (si, stage) in core.res_stages.iter().enumerate() {
                    let stage_hists = &mut res_hists[si];
                    let mut branch_sum = 0.0_f32;
                    let mut branch_n = 0usize;
                    for (g, kernel) in stage.kernels.iter().enumerate() {
                        let hist = &mut stage_hists[g];
                        if hist.is_empty() {
                            continue;
                        }
                        hist.rotate_right(1);
                        hist[0] = rs;
                        let mut yk = 0.0_f32;
                        for (i, kv) in kernel.iter().enumerate() {
                            yk += hist[i] * *kv;
                        }
                        let gain = stage.gains.get(g).copied().unwrap_or(0.25);
                        branch_sum += yk * gain;
                        branch_n += 1;
                    }
                    if branch_n > 0 {
                        rs = (rs + (branch_sum / branch_n as f32) * 0.03).tanh();
                    }
                }
                decoder_shaped = (decoder_shaped * 0.5 + rs * 0.5).tanh();
            }
        }

        // Multi-channel conditioned path (decoder-style coarse approximation).
        let mut ch_sum = 0.0_f32;
        for c in 0..8 {
            let alpha = (0.03 + 0.08 * (c as f32 / 7.0)).clamp(0.01, 0.2);
            band_lp[c] += alpha * (x - band_lp[c]);
            let band = if c % 2 == 0 {
                band_lp[c]
            } else {
                x - band_lp[c]
            };
            ch_sum += band * band_gains[c];
        }
        let cond_shaped = (ch_sum * 0.35).tanh();

        let transformed =
            (lp * formant_mix + tilt_hp * tilt + airy + decoder_shaped * 0.25 + cond_shaped * 0.35)
                .tanh()
                * drive.tanh().max(0.6);
        let out = x * (1.0 - identity_mix) + transformed * identity_mix;
        *s = out.clamp(-0.98, 0.98);
    }
}

fn unit_proxy(i: usize, hop: usize, y: &[f32]) -> [f32; 3] {
    let start = i.saturating_sub(hop / 2);
    let end = (start + hop.max(32)).min(y.len());
    let frame = &y[start..end];
    if frame.is_empty() {
        return [0.0, 0.0, 0.0];
    }
    let mean = crate::simd::sum(frame) / frame.len() as f32;
    let energy = crate::simd::sum_of_squares(frame) / frame.len() as f32;
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
    [mean, energy, zcr]
}

#[derive(Clone, Copy, Debug)]
struct StyleFrame {
    tilt: f32,
    air: f32,
    formant: f32,
    band_gains: [f32; 8],
}

impl Default for StyleFrame {
    fn default() -> Self {
        Self {
            tilt: 0.0,
            air: 0.0,
            formant: 1.0,
            band_gains: [0.0; 8],
        }
    }
}

fn neural_style_forward(
    core: &crate::burn_rvc::NeuralCore,
    unit: [f32; 3],
    f0_hz: f32,
) -> StyleFrame {
    let mut x768 = [0.0_f32; 768];
    for (i, x) in x768.iter_mut().enumerate() {
        let u = unit[i % 3];
        let h = ((i as f32 * 0.017).sin() * 0.5 + 0.5) * unit[(i + 1) % 3];
        *x = u * 0.7 + h * 0.3;
    }

    let mut z = [0.0_f32; 192];
    for (o, z_o) in z.iter_mut().enumerate() {
        let mut acc = core.emb_phone_b[o];
        let row = &core.emb_phone_w[o * 768..(o + 1) * 768];
        acc += crate::simd::dot_product(row, &x768);
        let pidx = (((f0_hz - 50.0) / (1200.0 - 50.0)) * 255.0)
            .round()
            .clamp(0.0, 255.0) as usize;
        acc += core.emb_pitch_w[pidx * 192 + o];
        *z_o = acc.tanh();
    }

    let spk_row = core.default_speaker.min(core.emb_g_rows.saturating_sub(1));
    let spk = &core.emb_g_w[spk_row * 256..(spk_row + 1) * 256];
    let mut cond = [0.0_f32; 512];
    for (o, c) in cond.iter_mut().enumerate() {
        let mut acc = core.cond_b[o];
        let row = &core.cond_w[o * 256..(o + 1) * 256];
        acc += crate::simd::dot_product(row, spk);
        *c = acc.tanh();
    }

    let m0 = crate::simd::sum(&z[..64]) / 64.0;
    let m1 = crate::simd::sum(&z[64..128]) / 64.0;
    let m2 = crate::simd::sum(&z[128..]) / 64.0;
    let c0 = crate::simd::sum(&cond[..64]) / 64.0;
    let c1 = crate::simd::sum(&cond[64..128]) / 64.0;
    let c2 = crate::simd::sum(&cond[128..192]) / 64.0;

    let tilt = (m1 + c1 * 0.4).tanh();
    let air = ((m0.abs() + c0.abs()) * 0.35).clamp(0.0, 1.0);
    let formant = (1.0 + (m2 + c2 * 0.5) * 0.25).clamp(0.7, 1.4);
    let mut band_gains = [0.0_f32; 8];
    for (i, gain) in band_gains.iter_mut().enumerate() {
        let start = i * 64;
        let end = start + 64;
        let mean = crate::simd::sum(&cond[start..end]) / 64.0;
        *gain = mean.tanh() * 0.9;
    }
    StyleFrame {
        tilt,
        air,
        formant,
        band_gains,
    }
}
