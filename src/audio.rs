use anyhow::{Context, Result, bail};
use std::io;
use std::path::Path;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use oxideav_core::{
    CodecId, CodecParameters, MediaType, Packet, RuntimeContext, SampleFormat, StreamInfo, TimeBase,
};

pub fn read_wav_mono(path: &Path) -> Result<(Vec<f32>, u32)> {
    let (samples, channels, sample_rate) =
        decode_wav_f32(path).with_context(|| format!("failed reading {}", path.display()))?;
    if channels != 1 {
        bail!("{} must be mono (has {channels} channels)", path.display());
    }
    Ok((samples, sample_rate))
}

pub fn write_wav_mono(path: &Path, x: &[f32], sr: u32) -> Result<()> {
    let bytes = pack_interleaved_samples_s16(x);

    let mut ctx = RuntimeContext::new();
    oxideav_basic::register(&mut ctx);

    let stream = audio_stream_info("pcm_s16le", 1, sr, SampleFormat::S16);
    let file = std::fs::File::create(path)
        .with_context(|| format!("failed creating {}", path.display()))?;
    let output: Box<dyn oxideav_core::WriteSeek> = Box::new(file);
    let mut mux = ctx
        .containers
        .open_muxer("wav", output, std::slice::from_ref(&stream))
        .map_err(oxideav_err_to_io)
        .with_context(|| format!("failed opening wav muxer for {}", path.display()))?;
    mux.write_header().map_err(oxideav_err_to_io)?;
    let packet = Packet::new(0, TimeBase::new(1, sr as i64), bytes);
    mux.write_packet(&packet).map_err(oxideav_err_to_io)?;
    mux.write_trailer().map_err(oxideav_err_to_io)?;
    Ok(())
}

fn decode_wav_f32(path: &Path) -> io::Result<(Vec<f32>, usize, u32)> {
    let file = std::fs::File::open(path)
        .map_err(|e| io::Error::other(format!("failed to open '{}': {e}", path.display())))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let format_opts = FormatOptions::default();
    let metadata_opts = MetadataOptions::default();
    let decoder_opts = DecoderOptions::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .map_err(|e| {
            io::Error::other(format!(
                "Symphonia failed to probe format for '{}': {e}",
                path.display()
            ))
        })?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .or_else(|| format.tracks().first())
        .ok_or_else(|| {
            io::Error::other(format!("No usable audio track in '{}'", path.display()))
        })?;

    let channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(1);
    let sample_rate = track.codec_params.sample_rate.unwrap_or(48_000);
    let track_id = track.id;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &decoder_opts)
        .map_err(|e| {
            io::Error::other(format!(
                "Symphonia failed to create decoder for '{}': {e}",
                path.display()
            ))
        })?;

    let mut sample_buf = None;
    let mut samples = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(e) => {
                return Err(io::Error::other(format!(
                    "Symphonia read error for '{}': {e}",
                    path.display()
                )));
            }
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = decoder.decode(&packet).map_err(|e| {
            io::Error::other(format!(
                "Symphonia decode error for '{}': {e}",
                path.display()
            ))
        })?;

        if sample_buf.is_none() {
            let spec = *decoded.spec();
            sample_buf = Some(SampleBuffer::<f32>::new(decoded.capacity() as u64, spec));
        }
        let buf = sample_buf.as_mut().unwrap();
        buf.copy_interleaved_ref(decoded);
        samples.extend_from_slice(buf.samples());
    }

    if samples.is_empty() {
        return Err(io::Error::other(format!(
            "No samples decoded from '{}'",
            path.display()
        )));
    }

    Ok((samples, channels, sample_rate))
}

fn pack_interleaved_samples_s16(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len().saturating_mul(2));
    for &sample in samples {
        let s = sample.clamp(-1.0, 1.0);
        let q = (s * i16::MAX as f32)
            .round()
            .clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        out.extend_from_slice(&q.to_le_bytes());
    }
    out
}

fn audio_codec_params(
    codec_id: &str,
    channels: usize,
    sample_rate: u32,
    sample_format: SampleFormat,
) -> CodecParameters {
    let mut params = CodecParameters::audio(CodecId::new(codec_id));
    params.media_type = MediaType::Audio;
    params.channels = Some(channels as u16);
    params.sample_rate = Some(sample_rate);
    params.sample_format = Some(sample_format);
    params
}

fn audio_stream_info(
    codec_id: &str,
    channels: usize,
    sample_rate: u32,
    sample_format: SampleFormat,
) -> StreamInfo {
    let params = audio_codec_params(codec_id, channels, sample_rate, sample_format);
    StreamInfo {
        index: 0,
        time_base: TimeBase::new(1, sample_rate as i64),
        duration: None,
        start_time: Some(0),
        params,
    }
}

fn oxideav_err_to_io(e: oxideav_core::Error) -> io::Error {
    io::Error::other(format!("OxideAV error: {e}"))
}

pub fn resample_linear(x: &[f32], in_sr: u32, out_sr: u32) -> Vec<f32> {
    if in_sr == out_sr || x.is_empty() {
        return x.to_vec();
    }
    let ratio = out_sr as f64 / in_sr as f64;
    let out_len = ((x.len() as f64) * ratio).round().max(1.0) as usize;
    let mut y = vec![0.0_f32; out_len];
    for (i, yi) in y.iter_mut().enumerate() {
        let t = (i as f64) / ratio;
        let i0 = t.floor() as usize;
        let i1 = (i0 + 1).min(x.len() - 1);
        let a = (t - i0 as f64) as f32;
        *yi = x[i0] * (1.0 - a) + x[i1] * a;
    }
    y
}

pub fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let s = crate::simd::sum_of_squares(x);
    (s / x.len() as f32).sqrt()
}

// Naive same-length pitch shift by variable-rate read.
// ratio > 1.0 raises pitch, ratio < 1.0 lowers pitch.
pub fn pitch_shift_linear_same_len(x: &[f32], ratio: f32) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    if (ratio - 1.0).abs() < 1.0e-6 {
        return x.to_vec();
    }
    let mut y = vec![0.0_f32; x.len()];
    let max_idx = (x.len() - 1) as f32;
    for (n, out) in y.iter_mut().enumerate() {
        // ratio > 1.0 should read faster through source => higher pitch.
        let src = (n as f32 * ratio).clamp(0.0, max_idx);
        let i0 = src.floor() as usize;
        let i1 = (i0 + 1).min(x.len() - 1);
        let a = src - i0 as f32;
        *out = x[i0] * (1.0 - a) + x[i1] * a;
    }
    y
}
