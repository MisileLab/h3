use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, SampleRate, Stream, StreamConfig};
use std::io::Cursor;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;

const SAMPLE_RATE: u32 = 16000;
const CHANNELS: u16 = 1;
const CHUNK_DURATION_SECS: f32 = 5.0;

pub struct AudioCapture {
    device: Device,
    config: StreamConfig,
    is_recording: Arc<AtomicBool>,
    stream: Option<Stream>,
}

impl AudioCapture {
    pub fn new() -> Result<Self, String> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or("No input device available")?;

        let device_name = device.name().unwrap_or_else(|_| "Unknown".to_string());
        println!("Using input device: {}", device_name);

        // Configure for 16kHz mono (optimal for Whisper)
        let config = StreamConfig {
            channels: CHANNELS,
            sample_rate: SampleRate(SAMPLE_RATE),
            buffer_size: cpal::BufferSize::Default,
        };

        Ok(Self {
            device,
            config,
            is_recording: Arc::new(AtomicBool::new(false)),
            stream: None,
        })
    }

    pub fn get_supported_config(&self) -> Result<StreamConfig, String> {
        let supported_configs = self
            .device
            .supported_input_configs()
            .map_err(|e| format!("Failed to get supported configs: {}", e))?;

        // Try to find a config that supports our desired sample rate
        for config in supported_configs {
            if config.channels() == CHANNELS
                && config.min_sample_rate().0 <= SAMPLE_RATE
                && config.max_sample_rate().0 >= SAMPLE_RATE
            {
                return Ok(StreamConfig {
                    channels: CHANNELS,
                    sample_rate: SampleRate(SAMPLE_RATE),
                    buffer_size: cpal::BufferSize::Default,
                });
            }
        }

        // Fall back to device default
        let default_config = self
            .device
            .default_input_config()
            .map_err(|e| format!("Failed to get default config: {}", e))?;

        Ok(StreamConfig {
            channels: default_config.channels(),
            sample_rate: default_config.sample_rate(),
            buffer_size: cpal::BufferSize::Default,
        })
    }

    pub fn start_capture(&mut self, audio_sender: mpsc::UnboundedSender<Vec<u8>>) -> Result<(), String> {
        if self.is_recording.load(Ordering::SeqCst) {
            return Err("Already recording".to_string());
        }

        let config = self.get_supported_config()?;
        let sample_rate = config.sample_rate.0;
        let channels = config.channels as usize;

        let samples_per_chunk = (sample_rate as f32 * CHUNK_DURATION_SECS) as usize * channels;
        let mut buffer: Vec<f32> = Vec::with_capacity(samples_per_chunk);

        let is_recording = self.is_recording.clone();
        is_recording.store(true, Ordering::SeqCst);

        let is_recording_clone = is_recording.clone();
        let audio_sender_clone = audio_sender.clone();

        // Check sample format
        let supported_config = self
            .device
            .default_input_config()
            .map_err(|e| format!("Failed to get config: {}", e))?;

        let stream = match supported_config.sample_format() {
            SampleFormat::F32 => self.build_stream::<f32>(
                &config,
                samples_per_chunk,
                buffer,
                is_recording_clone,
                audio_sender_clone,
                sample_rate,
                channels,
            )?,
            SampleFormat::I16 => self.build_stream_i16(
                &config,
                samples_per_chunk,
                is_recording_clone,
                audio_sender_clone,
                sample_rate,
                channels,
            )?,
            SampleFormat::U16 => {
                return Err("U16 sample format not supported".to_string());
            }
            _ => {
                return Err("Unsupported sample format".to_string());
            }
        };

        stream.play().map_err(|e| format!("Failed to start stream: {}", e))?;
        self.stream = Some(stream);

        println!("Audio capture started at {}Hz, {} channels", sample_rate, channels);
        Ok(())
    }

    fn build_stream<T>(
        &self,
        config: &StreamConfig,
        samples_per_chunk: usize,
        mut buffer: Vec<f32>,
        is_recording: Arc<AtomicBool>,
        audio_sender: mpsc::UnboundedSender<Vec<u8>>,
        sample_rate: u32,
        channels: usize,
    ) -> Result<Stream, String>
    where
        T: cpal::Sample + cpal::SizedSample + Into<f32>,
    {
        let err_fn = |err| eprintln!("Audio stream error: {}", err);

        let stream = self
            .device
            .build_input_stream(
                config,
                move |data: &[T], _: &cpal::InputCallbackInfo| {
                    if !is_recording.load(Ordering::SeqCst) {
                        return;
                    }

                    for &sample in data {
                        buffer.push(sample.into());

                        if buffer.len() >= samples_per_chunk {
                            // Convert to WAV and send
                            if let Ok(wav_data) = samples_to_wav(&buffer, sample_rate, channels as u16) {
                                let _ = audio_sender.send(wav_data);
                            }
                            buffer.clear();
                        }
                    }
                },
                err_fn,
                None,
            )
            .map_err(|e| format!("Failed to build stream: {}", e))?;

        Ok(stream)
    }

    fn build_stream_i16(
        &self,
        config: &StreamConfig,
        samples_per_chunk: usize,
        is_recording: Arc<AtomicBool>,
        audio_sender: mpsc::UnboundedSender<Vec<u8>>,
        sample_rate: u32,
        channels: usize,
    ) -> Result<Stream, String> {
        let mut buffer: Vec<f32> = Vec::with_capacity(samples_per_chunk);
        let err_fn = |err| eprintln!("Audio stream error: {}", err);

        let stream = self
            .device
            .build_input_stream(
                config,
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    if !is_recording.load(Ordering::SeqCst) {
                        return;
                    }

                    for &sample in data {
                        buffer.push(sample as f32 / i16::MAX as f32);

                        if buffer.len() >= samples_per_chunk {
                            if let Ok(wav_data) = samples_to_wav(&buffer, sample_rate, channels as u16) {
                                let _ = audio_sender.send(wav_data);
                            }
                            buffer.clear();
                        }
                    }
                },
                err_fn,
                None,
            )
            .map_err(|e| format!("Failed to build stream: {}", e))?;

        Ok(stream)
    }

    pub fn stop_capture(&mut self) {
        self.is_recording.store(false, Ordering::SeqCst);
        if let Some(stream) = self.stream.take() {
            drop(stream);
        }
        println!("Audio capture stopped");
    }

    pub fn is_recording(&self) -> bool {
        self.is_recording.load(Ordering::SeqCst)
    }
}

/// Convert f32 samples to WAV format bytes
fn samples_to_wav(samples: &[f32], sample_rate: u32, channels: u16) -> Result<Vec<u8>, String> {
    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec)
            .map_err(|e| format!("Failed to create WAV writer: {}", e))?;

        for &sample in samples {
            // Convert f32 [-1.0, 1.0] to i16
            let sample_i16 = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            writer
                .write_sample(sample_i16)
                .map_err(|e| format!("Failed to write sample: {}", e))?;
        }

        writer
            .finalize()
            .map_err(|e| format!("Failed to finalize WAV: {}", e))?;
    }

    Ok(cursor.into_inner())
}

/// List available input devices
pub fn list_input_devices() -> Vec<String> {
    let host = cpal::default_host();
    host.input_devices()
        .map(|devices| {
            devices
                .filter_map(|d| d.name().ok())
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_samples_to_wav() {
        let samples = vec![0.0f32; 1600]; // 100ms at 16kHz
        let result = samples_to_wav(&samples, 16000, 1);
        assert!(result.is_ok());
        let wav = result.unwrap();
        assert!(!wav.is_empty());
        // WAV header starts with "RIFF"
        assert_eq!(&wav[0..4], b"RIFF");
    }
}
