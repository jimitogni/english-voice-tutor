import {
  BookOpen,
  Brain,
  BriefcaseBusiness,
  ChevronDown,
  Plus,
  RotateCcw,
  SlidersHorizontal,
  Trash2,
  Volume2,
} from "lucide-react";
import { FormEvent, useState } from "react";
import type { ModeInfo, ObservabilitySummaryResponse, VadSettings, VoiceInfo } from "../types";
import { ObservabilityPanel } from "./ObservabilityPanel";

type VadSettingKey = "max_seconds" | "silence_seconds" | "energy_threshold" | "chunk_ms" | "min_speech_seconds";

interface VadControl {
  key: VadSettingKey;
  label: string;
  min: number;
  max: number;
  step: number;
  unit?: string;
  decimals: number;
}

const vadControls: VadControl[] = [
  {
    key: "max_seconds",
    label: "Max recording",
    min: 5,
    max: 300,
    step: 1,
    unit: "s",
    decimals: 0,
  },
  {
    key: "silence_seconds",
    label: "Silence stop",
    min: 0.3,
    max: 8,
    step: 0.1,
    unit: "s",
    decimals: 1,
  },
  {
    key: "energy_threshold",
    label: "Energy threshold",
    min: 0.001,
    max: 0.12,
    step: 0.001,
    decimals: 3,
  },
  {
    key: "chunk_ms",
    label: "VAD chunk",
    min: 10,
    max: 250,
    step: 10,
    unit: "ms",
    decimals: 0,
  },
  {
    key: "min_speech_seconds",
    label: "Min speech",
    min: 0.1,
    max: 3,
    step: 0.1,
    unit: "s",
    decimals: 1,
  },
];

interface SidebarProps {
  assistantName: string;
  userDisplayName: string;
  contextTurns: number;
  modes: ModeInfo[];
  selectedMode: string;
  onModeChange: (mode: string) => void;
  models: string[];
  installedModels: string[];
  selectedModel: string;
  onModelChange: (model: string) => void;
  selectedVoice: VoiceInfo | null;
  observabilitySummary: ObservabilitySummaryResponse | null;
  observabilityLoading: boolean;
  enableTts: boolean;
  onEnableTtsChange: (enabled: boolean) => void;
  focusWords: string[];
  focusWordsLimit: number;
  onAddFocusWord: (word: string) => Promise<void>;
  onRemoveFocusWord: (word: string) => Promise<void>;
  onReset: () => Promise<void>;
  vad: VadSettings;
  onVadChange: (settings: VadSettings) => void;
  busy: boolean;
}

function modeIcon(modeKey: string) {
  if (modeKey === "interview") {
    return <BriefcaseBusiness size={17} aria-hidden="true" />;
  }
  if (modeKey === "vocabulary") {
    return <BookOpen size={17} aria-hidden="true" />;
  }
  return <Brain size={17} aria-hidden="true" />;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function formattedVadValue(settings: VadSettings, control: VadControl): string {
  const value = settings[control.key];
  const formatted = value.toFixed(control.decimals);
  return control.unit ? `${formatted} ${control.unit}` : formatted;
}

function formattedDurationSummary(totalSeconds: number): string {
  const seconds = Math.max(0, Math.round(totalSeconds));
  if (seconds >= 60) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return remainingSeconds > 0 ? `${minutes}m ${remainingSeconds}s` : `${minutes} min`;
  }
  return `${seconds}s`;
}

export function Sidebar({
  assistantName,
  userDisplayName,
  contextTurns,
  modes,
  selectedMode,
  onModeChange,
  models,
  installedModels,
  selectedModel,
  onModelChange,
  selectedVoice,
  observabilitySummary,
  observabilityLoading,
  enableTts,
  onEnableTtsChange,
  focusWords,
  focusWordsLimit,
  onAddFocusWord,
  onRemoveFocusWord,
  onReset,
  vad,
  onVadChange,
  busy,
}: SidebarProps) {
  const [newFocusWord, setNewFocusWord] = useState("");
  const [focusBusy, setFocusBusy] = useState(false);
  const [vadOpen, setVadOpen] = useState(false);

  async function submitFocusWord(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const cleanWord = newFocusWord.trim();
    if (!cleanWord) {
      return;
    }
    setFocusBusy(true);
    try {
      await onAddFocusWord(cleanWord);
      setNewFocusWord("");
    } finally {
      setFocusBusy(false);
    }
  }

  const modelInstalled = installedModels.includes(selectedModel);

  function updateVad(control: VadControl, value: number) {
    if (!Number.isFinite(value)) {
      return;
    }
    const nextValue = clamp(value, control.min, control.max);
    onVadChange({
      ...vad,
      [control.key]: control.decimals === 0 ? Math.round(nextValue) : nextValue,
    });
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <span className="assistant-mark">J</span>
        <div>
          <h1>{assistantName}</h1>
          <p>{userDisplayName}</p>
        </div>
      </div>

      <section className="panel-section">
        <div className="section-heading">Mode</div>
        <div className="mode-list">
          {modes.map((mode) => (
            <button
              className={mode.key === selectedMode ? "mode-button active" : "mode-button"}
              key={mode.key}
              onClick={() => onModeChange(mode.key)}
              type="button"
            >
              {modeIcon(mode.key)}
              <span>{mode.label}</span>
            </button>
          ))}
        </div>
      </section>

      <section className="panel-section">
        <label className="field-label" htmlFor="model-select">
          Ollama Model
        </label>
        <select
          id="model-select"
          value={selectedModel}
          onChange={(event) => onModelChange(event.target.value)}
        >
          {models.map((model) => (
            <option key={model} value={model}>
              {model}
            </option>
          ))}
        </select>
        {!modelInstalled && (
          <p className="inline-warning">Run `ollama pull {selectedModel}` before using it.</p>
        )}
        {selectedVoice && (
          <div className="voice-card">
            <Volume2 size={16} aria-hidden="true" />
            <div>
              <strong>{selectedVoice.label}</strong>
              <span>{selectedVoice.available ? "Voice ready" : "Voice files missing"}</span>
            </div>
          </div>
        )}
        <label className="toggle-row">
          <input
            checked={enableTts}
            onChange={(event) => onEnableTtsChange(event.target.checked)}
            type="checkbox"
          />
          <span>Autoplay speech</span>
        </label>
      </section>

      <ObservabilityPanel loading={observabilityLoading} summary={observabilitySummary} />

      <section className="panel-section vad-section">
        <button
          aria-controls="voice-detection-settings"
          aria-expanded={vadOpen}
          className={vadOpen ? "section-heading vad-toggle open" : "section-heading vad-toggle"}
          onClick={() => setVadOpen((open) => !open)}
          type="button"
        >
          <span className="vad-toggle-title">
            <span className="status-light" aria-hidden="true" />
            <SlidersHorizontal size={15} aria-hidden="true" />
            <span>Voice Detection</span>
          </span>
          <span className="vad-toggle-summary">
            {formattedDurationSummary(vad.max_seconds)} max
            <ChevronDown size={15} aria-hidden="true" />
          </span>
        </button>
        {vadOpen && (
          <div className="vad-settings" id="voice-detection-settings">
            {vadControls.map((control) => {
              const value = Number(vad[control.key].toFixed(control.decimals));
              return (
                <label className="range-field" key={control.key}>
                  <span className="range-field-header">
                    <span>{control.label}</span>
                    <output>{formattedVadValue(vad, control)}</output>
                  </span>
                  <span className="range-inputs">
                    <input
                      aria-label={control.label}
                      max={control.max}
                      min={control.min}
                      onChange={(event) => updateVad(control, event.target.valueAsNumber)}
                      step={control.step}
                      type="range"
                      value={value}
                    />
                    <input
                      aria-label={`${control.label} value`}
                      className="number-input"
                      max={control.max}
                      min={control.min}
                      onChange={(event) => updateVad(control, event.target.valueAsNumber)}
                      step={control.step}
                      type="number"
                      value={value}
                    />
                  </span>
                </label>
              );
            })}
          </div>
        )}
      </section>

      <section className="panel-section focus-section">
        <div className="section-heading">
          Focus Words
          <span>{focusWords.length}/{focusWordsLimit}</span>
        </div>
        <form className="focus-form" onSubmit={submitFocusWord}>
          <input
            aria-label="Add focus word"
            disabled={focusBusy || focusWords.length >= focusWordsLimit}
            onChange={(event) => setNewFocusWord(event.target.value)}
            placeholder="word or expression"
            value={newFocusWord}
          />
          <button
            aria-label="Add focus word"
            disabled={focusBusy || focusWords.length >= focusWordsLimit}
            title="Add focus word"
            type="submit"
          >
            <Plus size={18} aria-hidden="true" />
          </button>
        </form>
        <div className="focus-list">
          {focusWords.map((word) => (
            <div className="focus-chip" key={word}>
              <span>{word}</span>
              <button
                aria-label={`Remove ${word}`}
                onClick={() => onRemoveFocusWord(word)}
                title="Remove word"
                type="button"
              >
                <Trash2 size={14} aria-hidden="true" />
              </button>
            </div>
          ))}
        </div>
      </section>

      <section className="panel-section session-section">
        <div>
          <span className="section-heading">Memory</span>
          <p>Last {contextTurns} turns</p>
        </div>
        <button className="icon-text-button" disabled={busy} onClick={onReset} type="button">
          <RotateCcw size={16} aria-hidden="true" />
          Reset
        </button>
      </section>
    </aside>
  );
}
