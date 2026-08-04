#[derive(Debug, Clone)]
pub struct FeatureIndex {
    pub bank: Vec<Vec<f32>>,
}

impl FeatureIndex {
    pub fn empty() -> Self {
        Self { bank: Vec::new() }
    }

    pub fn blend(&self, units: &[Vec<f32>], mix: f32) -> Vec<Vec<f32>> {
        if self.bank.is_empty() || mix <= 0.0 {
            return units.to_vec();
        }
        let a = mix.clamp(0.0, 1.0);
        units
            .iter()
            .map(|u| {
                let nn = nearest(u, &self.bank);
                u.iter()
                    .zip(nn.iter())
                    .map(|(x, y)| x * (1.0 - a) + y * a)
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

fn nearest<'a>(q: &[f32], bank: &'a [Vec<f32>]) -> &'a [f32] {
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for (i, v) in bank.iter().enumerate() {
        let d = l2(q, v);
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    &bank[best]
}

fn l2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut s = 0.0_f32;
    for i in 0..n {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}
