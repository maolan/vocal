use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use burn::tensor::DType;
use burn_store::{BurnpackStore, BurnpackWriter, ModuleStore, PytorchStore};

#[derive(Debug, Clone)]
pub struct BurnRvcTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

#[derive(Debug, Clone)]
pub struct BurnRvcModel {
    pub burnpack_path: PathBuf,
    pub tensors: Vec<BurnRvcTensor>,
    pub profile: RvcProfile,
    pub neural: Option<NeuralCore>,
}

#[derive(Debug, Clone, Copy)]
pub struct RvcProfile {
    pub brightness: f32,
    pub formant_shift: f32,
    #[allow(dead_code)]
    pub breath: f32,
    pub drive: f32,
    pub femininity: f32,
    pub speaker_strength: f32,
}

#[derive(Debug, Clone)]
pub struct NeuralCore {
    pub emb_phone_w: Vec<f32>, // [192, 768]
    pub emb_phone_b: Vec<f32>, // [192]
    pub emb_pitch_w: Vec<f32>, // [256, 192]
    pub emb_g_w: Vec<f32>,     // [N, 256]
    pub emb_g_rows: usize,
    pub cond_w: Vec<f32>, // [512, 256]
    pub cond_b: Vec<f32>, // [512]
    pub conv_pre_kernel: [f32; 7],
    pub conv_post_kernel: [f32; 7],
    pub up_kernels: Vec<GroupedKernelStage>,
    pub res_stages: Vec<ResKernelStage>,
    pub default_speaker: usize,
}

#[derive(Debug, Clone)]
pub struct GroupedKernelStage {
    pub kernels: Vec<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct ResKernelStage {
    pub kernels: Vec<Vec<f32>>,
    pub gains: Vec<f32>,
}

impl BurnRvcModel {
    pub fn from_burnpack(path: impl AsRef<Path>) -> Result<Self> {
        let burnpack_path = path.as_ref().to_path_buf();
        let mut store = BurnpackStore::from_file(&burnpack_path).zero_copy(true);
        let snapshots = store
            .get_all_snapshots()
            .with_context(|| format!("failed reading burnpack {}", burnpack_path.display()))?;

        let mut tensors = Vec::with_capacity(snapshots.len());
        let mut sample_values = Vec::new();
        let mut speaker_values = Vec::new();
        let mut speaker_tensor_hits = 0usize;
        for (name, snapshot) in snapshots {
            let data = snapshot
                .to_data()
                .with_context(|| format!("failed decoding tensor {}", name))?;
            let lower = name.to_ascii_lowercase();
            let is_speaker_tensor = lower.contains("emb_g")
                || lower.contains("speaker")
                || lower.contains("spk")
                || lower.contains("sid")
                || lower.contains("text_enc")
                || lower.contains("enc_p");
            if sample_values.len() < 256 || (is_speaker_tensor && speaker_values.len() < 256) {
                let converted = data.clone().convert_dtype(DType::F32);
                if let Ok(vals) = converted.to_vec::<f32>() {
                    for v in vals.into_iter().take(16) {
                        if sample_values.len() < 256 && v.is_finite() {
                            sample_values.push(v);
                        }
                        if is_speaker_tensor && speaker_values.len() < 256 && v.is_finite() {
                            speaker_values.push(v);
                        }
                        if sample_values.len() >= 256
                            && (!is_speaker_tensor || speaker_values.len() >= 256)
                        {
                            break;
                        }
                    }
                }
            }
            if is_speaker_tensor {
                speaker_tensor_hits += 1;
            }
            tensors.push(BurnRvcTensor {
                name: name.clone(),
                shape: data.shape.to_vec(),
                dtype: data.dtype,
            });
        }

        let profile = build_profile(&sample_values, &speaker_values, speaker_tensor_hits);
        let neural = build_neural_core(snapshots);

        Ok(Self {
            burnpack_path,
            tensors,
            profile,
            neural,
        })
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn summary(&self) -> String {
        let f32_count = self
            .tensors
            .iter()
            .filter(|t| matches!(t.dtype, DType::F32))
            .count();
        format!(
            "burnpack={} tensors={} f32_tensors={}",
            self.burnpack_path.display(),
            self.tensor_count(),
            f32_count
        )
    }

    pub fn first_tensor_name(&self) -> Option<&str> {
        self.tensors.first().map(|t| t.name.as_str())
    }

    pub fn dump_tensors(&self, limit: usize) -> Vec<String> {
        self.tensors
            .iter()
            .take(limit)
            .map(|t| format!("{}\t{:?}\t{:?}", t.name, t.dtype, t.shape))
            .collect()
    }
}

fn build_profile(values: &[f32], speaker_values: &[f32], speaker_tensor_hits: usize) -> RvcProfile {
    let mut acc = [0.0_f32; 4];
    if values.is_empty() {
        return RvcProfile {
            brightness: 0.5,
            formant_shift: 1.0,
            breath: 0.2,
            drive: 0.2,
            femininity: 0.5,
            speaker_strength: 0.0,
        };
    }
    for (i, v) in values.iter().enumerate() {
        let t = (v.abs().ln_1p() * 0.5).clamp(0.0, 1.0);
        acc[i % 4] += t;
    }
    let n = (values.len() as f32 / 4.0).max(1.0);
    let a0 = (acc[0] / n).clamp(0.0, 1.0);
    let a1 = (acc[1] / n).clamp(0.0, 1.0);
    let a2 = (acc[2] / n).clamp(0.0, 1.0);
    let a3 = (acc[3] / n).clamp(0.0, 1.0);
    let feminine_base = if speaker_values.is_empty() {
        0.5
    } else {
        let mean_abs =
            speaker_values.iter().map(|v| v.abs()).sum::<f32>() / speaker_values.len() as f32;
        (mean_abs.ln_1p() * 0.45).clamp(0.0, 1.0)
    };
    let speaker_strength = ((speaker_tensor_hits as f32) / 64.0).clamp(0.0, 1.0);
    RvcProfile {
        brightness: a0,
        formant_shift: 0.8 + a1 * 0.6,
        breath: a2 * 0.4,
        drive: a3 * 0.6,
        femininity: feminine_base,
        speaker_strength,
    }
}

fn build_neural_core(
    snapshots: &std::collections::BTreeMap<String, burn_store::TensorSnapshot>,
) -> Option<NeuralCore> {
    let (emb_phone_w, emb_phone_w_shape) =
        get_f32_tensor(snapshots, "weight.enc_p.emb_phone.weight")?;
    let (emb_phone_b, emb_phone_b_shape) =
        get_f32_tensor(snapshots, "weight.enc_p.emb_phone.bias")?;
    let (emb_pitch_w, emb_pitch_w_shape) =
        get_f32_tensor(snapshots, "weight.enc_p.emb_pitch.weight")?;
    let (emb_g_w, emb_g_w_shape) = get_f32_tensor(snapshots, "weight.emb_g.weight")?;
    let (cond_w, cond_w_shape) = get_f32_tensor(snapshots, "weight.dec.cond.weight")?;
    let (cond_b, cond_b_shape) = get_f32_tensor(snapshots, "weight.dec.cond.bias")?;
    let (conv_pre_w, conv_pre_w_shape) = get_f32_tensor(snapshots, "weight.dec.conv_pre.weight")?;
    let (conv_post_w, conv_post_w_shape) =
        get_f32_tensor(snapshots, "weight.dec.conv_post.weight")?;

    if emb_phone_w_shape != vec![192, 768]
        || emb_phone_b_shape != vec![192]
        || emb_pitch_w_shape != vec![256, 192]
        || cond_b_shape != vec![512]
        || cond_w_shape.len() != 3
        || cond_w_shape[0] != 512
        || cond_w_shape[1] != 256
        || conv_pre_w_shape != vec![512, 192, 7]
        || conv_post_w_shape != vec![1, 32, 7]
    {
        return None;
    }

    let emb_g_rows = *emb_g_w_shape.first()?;
    if emb_g_w_shape.len() != 2 || emb_g_w_shape[1] != 256 || emb_g_rows == 0 {
        return None;
    }

    let mut best_row = 0usize;
    let mut best_score = f32::MIN;
    for r in 0..emb_g_rows {
        let row = &emb_g_w[r * 256..(r + 1) * 256];
        let mean = crate::simd::sum(row) / 256.0;
        let energy = crate::simd::sum_of_squares(row) / 256.0;
        let score = mean.abs() + energy.sqrt();
        if score > best_score {
            best_score = score;
            best_row = r;
        }
    }

    let mut conv_pre_kernel = [0.0_f32; 7];
    for (k, out) in conv_pre_kernel.iter_mut().enumerate() {
        let mut sum = 0.0_f32;
        let mut count = 0usize;
        for o in 0..512 {
            for i in 0..192 {
                let idx = (o * 192 + i) * 7 + k;
                sum += conv_pre_w[idx];
                count += 1;
            }
        }
        *out = if count > 0 { sum / count as f32 } else { 0.0 };
    }
    // The inner loops above iterate over strided elements (every 7th), which
    // is not contiguous and thus not SIMD-friendly; leave scalar.

    let mut conv_post_kernel = [0.0_f32; 7];
    for (k, out) in conv_post_kernel.iter_mut().enumerate() {
        let mut sum = 0.0_f32;
        for i in 0..32 {
            let idx = i * 7 + k;
            sum += conv_post_w[idx];
        }
        *out = sum / 32.0;
    }

    let mut up_kernels = Vec::new();
    for up_idx in 0..4 {
        let key = format!("weight.dec.ups.{up_idx}.weight_v");
        if let Some((w, shape)) = get_f32_tensor(snapshots, &key)
            && shape.len() == 3
        {
            let ksz = shape[2];
            if ksz > 0 {
                let groups = 8usize.min(shape[0].max(1));
                let mut kernels = vec![vec![0.0_f32; ksz]; groups];
                for (g, gk) in kernels.iter_mut().enumerate() {
                    let oc_start = g * shape[0] / groups;
                    let oc_end = ((g + 1) * shape[0] / groups).max(oc_start + 1);
                    for (k, out) in gk.iter_mut().enumerate() {
                        let mut sum = 0.0_f32;
                        let mut count = 0usize;
                        for oc in oc_start..oc_end {
                            for ic in 0..shape[1] {
                                let idx = (oc * shape[1] + ic) * ksz + k;
                                sum += w[idx];
                                count += 1;
                            }
                        }
                        *out = if count > 0 { sum / count as f32 } else { 0.0 };
                    }
                }
                up_kernels.push(GroupedKernelStage { kernels });
            }
        }
    }

    let mut res_stages = Vec::new();
    for rb in 0..12 {
        for convs in [1, 2] {
            for li in 0..3 {
                let v_key = format!("weight.dec.resblocks.{rb}.convs{convs}.{li}.weight_v");
                let g_key = format!("weight.dec.resblocks.{rb}.convs{convs}.{li}.weight_g");
                let Some((v, shape)) = get_f32_tensor(snapshots, &v_key) else {
                    continue;
                };
                if shape.len() != 3 || shape[2] == 0 {
                    continue;
                }
                let ksz = shape[2];
                let groups = 8usize.min(shape[0].max(1));
                let mut kernels = vec![vec![0.0_f32; ksz]; groups];
                let mut gains = vec![0.25_f32; groups];
                for (gi, gk) in kernels.iter_mut().enumerate() {
                    let oc_start = gi * shape[0] / groups;
                    let oc_end = ((gi + 1) * shape[0] / groups).max(oc_start + 1);
                    for (k, out) in gk.iter_mut().enumerate() {
                        let mut sum = 0.0_f32;
                        let mut count = 0usize;
                        for oc in oc_start..oc_end {
                            for ic in 0..shape[1] {
                                let idx = (oc * shape[1] + ic) * ksz + k;
                                sum += v[idx];
                                count += 1;
                            }
                        }
                        *out = if count > 0 { sum / count as f32 } else { 0.0 };
                    }
                }
                if let Some((g, gshape)) = get_f32_tensor(snapshots, &g_key)
                    && !gshape.is_empty()
                {
                    for (gi, out) in gains.iter_mut().enumerate() {
                        let start = gi * g.len() / groups;
                        let end = ((gi + 1) * g.len() / groups).max(start + 1);
                        let mut sum = 0.0_f32;
                        for v in &g[start..end] {
                            sum += v.abs();
                        }
                        let mean = sum / (end - start) as f32;
                        *out = mean.clamp(0.02, 2.0);
                    }
                }
                res_stages.push(ResKernelStage { kernels, gains });
            }
        }
    }

    Some(NeuralCore {
        emb_phone_w,
        emb_phone_b,
        emb_pitch_w,
        emb_g_w,
        emb_g_rows,
        cond_w: cond_w.into_iter().take(512 * 256).collect(),
        cond_b,
        conv_pre_kernel,
        conv_post_kernel,
        up_kernels,
        res_stages,
        default_speaker: best_row,
    })
}

fn get_f32_tensor(
    snapshots: &std::collections::BTreeMap<String, burn_store::TensorSnapshot>,
    name: &str,
) -> Option<(Vec<f32>, Vec<usize>)> {
    let snapshot = snapshots.get(name)?;
    let data = snapshot.to_data().ok()?;
    let shape = data.shape.to_vec();
    let converted = data.convert_dtype(DType::F32);
    let vals = converted.to_vec::<f32>().ok()?;
    Some((vals, shape))
}

pub fn convert_pth_to_burnpack(
    pth_path: impl AsRef<Path>,
    output_bpk: impl AsRef<Path>,
    top_level_key: Option<&str>,
) -> Result<usize> {
    let pth_path = pth_path.as_ref();
    let output_bpk = output_bpk.as_ref();

    let mut pytorch_store = match top_level_key {
        Some(key) if !key.is_empty() => PytorchStore::from_file(pth_path).with_top_level_key(key),
        _ => PytorchStore::from_file(pth_path),
    };
    let snapshots = pytorch_store
        .get_all_snapshots()
        .with_context(|| format!("failed reading pth {}", pth_path.display()))?;

    let tensors: Vec<_> = snapshots.values().cloned().collect();
    let tensor_count = tensors.len();
    BurnpackWriter::new(tensors)
        .with_metadata("source_format", "pytorch")
        .with_metadata("source_path", &pth_path.display().to_string())
        .write_to_file(output_bpk)
        .with_context(|| format!("failed writing burnpack {}", output_bpk.display()))?;

    Ok(tensor_count)
}
