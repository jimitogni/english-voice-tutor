import { BookOpen, Brain, BriefcaseBusiness, Plus, RotateCcw, Trash2, Volume2 } from "lucide-react";
import { FormEvent, useState } from "react";
import type { ModeInfo, VoiceInfo } from "../types";

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
  enableTts: boolean;
  onEnableTtsChange: (enabled: boolean) => void;
  focusWords: string[];
  focusWordsLimit: number;
  onAddFocusWord: (word: string) => Promise<void>;
  onRemoveFocusWord: (word: string) => Promise<void>;
  onReset: () => Promise<void>;
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
  enableTts,
  onEnableTtsChange,
  focusWords,
  focusWordsLimit,
  onAddFocusWord,
  onRemoveFocusWord,
  onReset,
  busy,
}: SidebarProps) {
  const [newFocusWord, setNewFocusWord] = useState("");
  const [focusBusy, setFocusBusy] = useState(false);

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
