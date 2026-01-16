import type { SwToOffscreenMessage } from './types';

let audioContext: AudioContext | null = null;
let audioWorkletNode: AudioWorkletNode | null = null;
let source: MediaStreamAudioSourceNode | null = null;
let audioStream: MediaStream | null = null;

chrome.runtime.onConnect.addListener((port: chrome.runtime.Port) => {
  if (port.name === 'sw') {
    port.onMessage.addListener(async (message: SwToOffscreenMessage) => {
      switch (message.type) {
        case 'START_AUDIO':
          await startAudio(message.streamId);
          break;
        case 'STOP_AUDIO':
          stopAudio();
          break;
        case 'RESUME_AUDIO':
          resumeAudio();
          break;
      }
    });
  }
});

async function startAudio(streamId: string): Promise<void> {
  try {
    // Get the media stream from the stream ID
    audioStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        // @ts-expect-error - Chrome-specific constraint for tab capture
        mandatory: {
          chromeMediaSource: 'tab',
          chromeMediaSourceId: streamId,
        },
      },
    });

    if (!audioStream) {
      throw new Error('Failed to get media stream');
    }

    audioContext = new AudioContext({
      sampleRate: 24000,
    });

    source = audioContext.createMediaStreamSource(audioStream);

    await audioContext.audioWorklet.addModule('dist/worklet-processor.js');
    audioWorkletNode = new AudioWorkletNode(audioContext, 'audio-processor');

    source.connect(audioWorkletNode);

    const swPort = chrome.runtime.connect({ name: 'offscreen' });

    audioWorkletNode.port.onmessage = (event: MessageEvent) => {
      if (event.data.type === 'AUDIO_FRAME') {
        swPort.postMessage({
          type: 'AUDIO_FRAME',
          data: event.data.data,
        });
      }
    };

    audioWorkletNode.port.postMessage({
      type: 'INIT',
      sampleRate: 24000,
      frameSize: 480,
    });
  } catch (error) {
    console.error('Failed to start audio:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    chrome.runtime.sendMessage({
      type: 'STATUS_UPDATE',
      status: 'error',
      message: `Audio error: ${errorMessage}`,
    });
  }
}

function stopAudio(): void {
  if (source) {
    source.disconnect();
    source = null;
  }

  if (audioWorkletNode) {
    audioWorkletNode.disconnect();
    audioWorkletNode = null;
  }

  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }

  if (audioStream) {
    audioStream.getTracks().forEach((track) => track.stop());
    audioStream = null;
  }
}

function resumeAudio(): void {
  if (audioContext && audioContext.state === 'suspended') {
    audioContext.resume();
  }
}
