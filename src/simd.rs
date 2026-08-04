#![allow(unsafe_op_in_unsafe_fn)]

use std::arch::x86_64::*;
use wide::f32x4;

pub fn mul_inplace(buf: &mut [f32], gain: f32) {
    if gain == 1.0 || buf.is_empty() {
        return;
    }
    if is_x86_feature_detected!("avx") {
        unsafe { mul_inplace_avx(buf, gain) }
    } else if is_x86_feature_detected!("sse") {
        unsafe { mul_inplace_sse(buf, gain) }
    } else {
        for v in buf {
            *v *= gain;
        }
    }
}

#[target_feature(enable = "avx")]
unsafe fn mul_inplace_avx(buf: &mut [f32], gain: f32) {
    let g = _mm256_set1_ps(gain);
    let n = buf.len() / 8;
    for i in 0..n {
        let chunk = &mut buf[i * 8..(i + 1) * 8];
        let x = _mm256_loadu_ps(chunk.as_ptr());
        _mm256_storeu_ps(chunk.as_mut_ptr(), _mm256_mul_ps(x, g));
    }
    for v in &mut buf[n * 8..] {
        *v *= gain;
    }
}

#[target_feature(enable = "sse")]
unsafe fn mul_inplace_sse(buf: &mut [f32], gain: f32) {
    let n = buf.len() / 4;
    let g: f32x4 = [gain; 4].into();
    for i in 0..n {
        let chunk = &mut buf[i * 4..(i + 1) * 4];
        let x: f32x4 = [chunk[0], chunk[1], chunk[2], chunk[3]].into();
        let r = x * g;
        chunk.copy_from_slice(&r.to_array());
    }
    for v in &mut buf[n * 4..] {
        *v *= gain;
    }
}

pub fn sum(buf: &[f32]) -> f32 {
    if buf.is_empty() {
        return 0.0;
    }
    if is_x86_feature_detected!("avx") {
        unsafe { sum_avx(buf) }
    } else if is_x86_feature_detected!("sse") {
        unsafe { sum_sse(buf) }
    } else {
        buf.iter().copied().sum()
    }
}

#[target_feature(enable = "avx")]
unsafe fn sum_avx(buf: &[f32]) -> f32 {
    let n = buf.len() / 8;
    let mut acc = _mm256_setzero_ps();
    for i in 0..n {
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(buf[i * 8..].as_ptr()));
    }
    let mut s = hsum256_ps(acc);
    for &v in &buf[n * 8..] {
        s += v;
    }
    s
}

#[target_feature(enable = "sse")]
unsafe fn sum_sse(buf: &[f32]) -> f32 {
    let n = buf.len() / 4;
    let mut acc = f32x4::ZERO;
    for i in 0..n {
        let x: f32x4 = f32x4::from(&buf[i * 4..(i + 1) * 4]);
        acc += x;
    }
    let mut s = acc.to_array().iter().copied().sum::<f32>();
    for &v in &buf[n * 4..] {
        s += v;
    }
    s
}

pub fn sum_of_squares(buf: &[f32]) -> f32 {
    if buf.is_empty() {
        return 0.0;
    }
    if is_x86_feature_detected!("avx") {
        unsafe { sum_of_squares_avx(buf) }
    } else if is_x86_feature_detected!("sse") {
        unsafe { sum_of_squares_sse(buf) }
    } else {
        buf.iter().map(|v| v * v).sum()
    }
}

#[target_feature(enable = "avx")]
unsafe fn sum_of_squares_avx(buf: &[f32]) -> f32 {
    let n = buf.len() / 8;
    let mut acc = _mm256_setzero_ps();
    for i in 0..n {
        let x = _mm256_loadu_ps(buf[i * 8..].as_ptr());
        acc = _mm256_add_ps(acc, _mm256_mul_ps(x, x));
    }
    let mut s = hsum256_ps(acc);
    for &v in &buf[n * 8..] {
        s += v * v;
    }
    s
}

#[target_feature(enable = "sse")]
unsafe fn sum_of_squares_sse(buf: &[f32]) -> f32 {
    let n = buf.len() / 4;
    let mut acc = f32x4::ZERO;
    for i in 0..n {
        let x: f32x4 = f32x4::from(&buf[i * 4..(i + 1) * 4]);
        acc += x * x;
    }
    let mut s = acc.to_array().iter().copied().sum::<f32>();
    for &v in &buf[n * 4..] {
        s += v * v;
    }
    s
}

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let a = &a[..n];
    let b = &b[..n];
    if is_x86_feature_detected!("avx") {
        unsafe { dot_product_avx(a, b) }
    } else if is_x86_feature_detected!("sse") {
        unsafe { dot_product_sse(a, b) }
    } else {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }
}

#[target_feature(enable = "avx")]
unsafe fn dot_product_avx(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() / 8;
    let mut acc = _mm256_setzero_ps();
    for i in 0..n {
        let x = _mm256_loadu_ps(a[i * 8..].as_ptr());
        let y = _mm256_loadu_ps(b[i * 8..].as_ptr());
        acc = _mm256_add_ps(acc, _mm256_mul_ps(x, y));
    }
    let mut s = hsum256_ps(acc);
    for i in n * 8..a.len() {
        s += a[i] * b[i];
    }
    s
}

#[target_feature(enable = "sse")]
unsafe fn dot_product_sse(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() / 4;
    let mut acc = f32x4::ZERO;
    for i in 0..n {
        let x: f32x4 = f32x4::from(&a[i * 4..(i + 1) * 4]);
        let y: f32x4 = f32x4::from(&b[i * 4..(i + 1) * 4]);
        acc += x * y;
    }
    let mut s = acc.to_array().iter().copied().sum::<f32>();
    for i in n * 4..a.len() {
        s += a[i] * b[i];
    }
    s
}

#[target_feature(enable = "avx")]
unsafe fn hsum256_ps(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehl_ps(s128, s128);
    let sums = _mm_add_ps(s128, shuf);
    let shuf2 = _mm_shuffle_ps(sums, sums, 0x01);
    let final_sum = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(final_sum)
}
