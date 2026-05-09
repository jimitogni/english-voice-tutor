import { FormEvent, useEffect, useRef, useState } from "react";
import { Mic, Send, Square } from "lucide-react";
import type { VadSettings } from "../types";

interface VoiceDockProps {
  busy: boolean;
  vad: VadSettings | null;
  onSendText: (message: string) => Promise<void>;
  onSendAudio: (audio: Blob) => Promise<void>;
}

function preferredMimeType(): string {
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
  return candidates.find((candidate) => MediaRecorder.isTypeSupported(candidate)) ?? "";
}

export function VoiceDock({ busy, vad, onSendText, onSendAudio }: VoiceDockProps) {
  const [text, setText] = useState("");
  const [recording, setRecording] = useState(false);
  const [recordingError, setRecordingError] = useState<string | null>(null);
  const [level, setLevel] = useState(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationRef = useRef<number | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const startedAtRef = useRef<number>(0);
  const speechStartedAtRef = useRef<number | null>(null);
  const silenceStartedAtRef = useRef<number | null>(null);
  const stoppingRef = useRef(false);
  const onSendAudioRef = useRef(onSendAudio);
  const vadRef = useRef(vad);

  useEffect(() => {
    onSendAudioRef.current = onSendAudio;
  }, [onSendAudio]);

  useEffect(() => {
    vadRef.current = vad;
  }, [vad]);

  useEffect(() => {
    return () => {
      stopTracks();
      cancelAnimation();
      void audioContextRef.current?.close();
    };
  }, []);

  function cancelAnimation() {
    if (animationRef.current !== null) {
      window.cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
  }

  function stopTracks() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }

  function stopRecording() {
    if (stoppingRef.current) {
      return;
    }
    stoppingRef.current = true;
    cancelAnimation();
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
      return;
    }
    stopTracks();
    setRecording(false);
  }

  function readLevel(samples: Float32Array): number {
    let sum = 0;
    for (const sample of samples) {
      sum += sample * sample;
    }
    return Math.sqrt(sum / samples.length);
  }

  function monitorSilence() {
    const analyser = analyserRef.current;
    const settings = vadRef.current;
    if (!analyser || !settings) {
      return;
    }

    const samples = new Float32Array(analyser.fftSize);
    const tick = () => {
      analyser.getFloatTimeDomainData(samples);
      const rms = readLevel(samples);
      setLevel(Math.min(1, rms / Math.max(settings.energy_threshold * 3, 0.001)));

      const now = performance.now();
      const elapsedSeconds = (now - startedAtRef.current) / 1000;
      const speechStartedAt = speechStartedAtRef.current;
      const hasVoice = rms >= settings.energy_threshold;

      if (hasVoice) {
        if (speechStartedAt === null) {
          speechStartedAtRef.current = now;
        }
        silenceStartedAtRef.current = null;
      } else if (
        speechStartedAt !== null &&
        (now - speechStartedAt) / 1000 >= settings.min_speech_seconds
      ) {
        if (silenceStartedAtRef.current === null) {
          silenceStartedAtRef.current = now;
        }
        if ((now - silenceStartedAtRef.current) / 1000 >= settings.silence_seconds) {
          stopRecording();
          return;
        }
      }

      if (elapsedSeconds >= settings.max_seconds) {
        stopRecording();
        return;
      }

      animationRef.current = window.requestAnimationFrame(tick);
    };

    animationRef.current = window.requestAnimationFrame(tick);
  }

  async function startRecording() {
    if (!vad) {
      setRecordingError("Voice settings are not loaded yet.");
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setRecordingError("This browser does not expose microphone recording for localhost.");
      return;
    }
    if (!window.MediaRecorder) {
      setRecordingError("This browser does not support MediaRecorder.");
      return;
    }

    setRecordingError(null);
    stoppingRef.current = false;
    speechStartedAtRef.current = null;
    silenceStartedAtRef.current = null;
    chunksRef.current = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: vad.sample_rate,
        },
      });
      streamRef.current = stream;

      const mimeType = preferredMimeType();
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" });
        stopTracks();
        void audioContextRef.current?.close();
        audioContextRef.current = null;
        analyserRef.current = null;
        setRecording(false);
        setLevel(0);
        if (blob.size > 0) {
          void onSendAudioRef.current(blob);
        }
      };
      recorder.onerror = () => {
        setRecordingError("The browser recorder failed. Try again or use typed input.");
        stopRecording();
      };

      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass) {
        setRecordingError("This browser does not support Web Audio recording analysis.");
        stopRecording();
        return;
      }
      const audioContext = new AudioContextClass({ sampleRate: vad.sample_rate });
      audioContextRef.current = audioContext;
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);
      analyserRef.current = analyser;

      startedAtRef.current = performance.now();
      recorder.start(250);
      setRecording(true);
      monitorSilence();
    } catch (error) {
      stopTracks();
      setRecording(false);
      setRecordingError(
        error instanceof Error
          ? error.message
          : "Could not access the microphone. Check browser permissions.",
      );
    }
  }

  async function submitText(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const cleanText = text.trim();
    if (!cleanText || busy) {
      return;
    }
    setText("");
    await onSendText(cleanText);
  }

  return (
    <footer className="voice-dock">
      <div className="recording-controls">
        <button
          className={recording ? "record-button recording" : "record-button"}
          disabled={busy}
          onClick={recording ? stopRecording : startRecording}
          title={recording ? "Stop recording" : "Start recording"}
          type="button"
        >
          {recording ? <Square size={22} aria-hidden="true" /> : <Mic size={22} aria-hidden="true" />}
        </button>
        <div className="level-meter" aria-hidden="true">
          <span style={{ transform: `scaleX(${level})` }} />
        </div>
      </div>

      <form className="text-composer" onSubmit={submitText}>
        <input
          disabled={busy || recording}
          onChange={(event) => setText(event.target.value)}
          placeholder="Type a message"
          value={text}
        />
        <button aria-label="Send typed message" disabled={busy || recording || !text.trim()} title="Send" type="submit">
          <Send size={20} aria-hidden="true" />
        </button>
      </form>

      {recordingError && <div className="dock-error">{recordingError}</div>}
    </footer>
  );
}

declare global {
  interface Window {
    webkitAudioContext?: typeof AudioContext;
  }
}
