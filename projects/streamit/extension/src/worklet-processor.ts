// AudioWorklet processor for audio processing
// Note: This file runs in a separate AudioWorklet context

interface AudioWorkletProcessorInterface {
  readonly port: MessagePort;
  process(
    inputs: Float32Array[][],
    outputs: Float32Array[][],
    parameters: Record<string, Float32Array>
  ): boolean;
}

declare class AudioWorkletProcessor implements AudioWorkletProcessorInterface {
  readonly port: MessagePort;
  constructor();
  process(
    inputs: Float32Array[][],
    outputs: Float32Array[][],
    parameters: Record<string, Float32Array>
  ): boolean;
}

declare function registerProcessor(
  name: string,
  processorCtor: new () => AudioWorkletProcessor
): void;

class StreamItAudioProcessor extends AudioWorkletProcessor {
  private inputSampleRate: number | null = null;
  private targetSampleRate = 24000;
  private frameSize = 480;
  private buffer: Float32Array = new Float32Array(0);

  constructor() {
    super();
  }

  process(
    inputs: Float32Array[][],
    _outputs: Float32Array[][],
    _parameters: Record<string, Float32Array>
  ): boolean {
    const input = inputs[0];

    if (!input || !input[0]) {
      return true;
    }

    const inputData = input[0];

    if (this.inputSampleRate === null) {
      // Request sample rate from main thread
      this.port.postMessage({
        type: 'INIT',
      });
      // Use a default until we get actual sample rate
      this.inputSampleRate = 48000;
    }

    const resampled = this.resample(inputData);

    // Append to buffer
    const newBuffer = new Float32Array(this.buffer.length + resampled.length);
    newBuffer.set(this.buffer);
    newBuffer.set(resampled, this.buffer.length);
    this.buffer = newBuffer;

    while (this.buffer.length >= this.frameSize) {
      const frame = this.buffer.slice(0, this.frameSize);
      this.buffer = this.buffer.slice(this.frameSize);

      const pcm16 = this.floatToPCM16(frame);
      this.port.postMessage(
        {
          type: 'AUDIO_FRAME',
          data: pcm16.buffer,
        },
        [pcm16.buffer]
      );
    }

    return true;
  }

  private resample(inputData: Float32Array): Float32Array {
    if (this.inputSampleRate === this.targetSampleRate) {
      return new Float32Array(inputData);
    }

    const ratio = this.inputSampleRate! / this.targetSampleRate;
    const outputLength = Math.floor(inputData.length / ratio);
    const output = new Float32Array(outputLength);

    for (let i = 0; i < outputLength; i++) {
      const srcIndex = i * ratio;
      const srcIndexFloor = Math.floor(srcIndex);
      const srcIndexCeil = Math.min(srcIndexFloor + 1, inputData.length - 1);
      const frac = srcIndex - srcIndexFloor;

      output[i] = inputData[srcIndexFloor] * (1 - frac) + inputData[srcIndexCeil] * frac;
    }

    return output;
  }

  private floatToPCM16(floatData: Float32Array): Int16Array {
    const pcm16 = new Int16Array(floatData.length);

    for (let i = 0; i < floatData.length; i++) {
      let sample = floatData[i];
      sample = Math.max(-1, Math.min(1, sample));
      pcm16[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    }

    return pcm16;
  }
}

registerProcessor('audio-processor', StreamItAudioProcessor);
