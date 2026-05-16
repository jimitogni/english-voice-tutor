import { useEffect, useMemo, useState } from "react";
import {
  addFocusWord,
  fetchObservabilitySummary,
  fetchStatus,
  removeFocusWord,
  resetSession,
  sendChat,
  sendVoice,
} from "./api";
import { ChatWindow } from "./components/ChatWindow";
import { ObservabilityPage } from "./components/ObservabilityPage";
import { Sidebar } from "./components/Sidebar";
import { VoiceDock } from "./components/VoiceDock";
import type {
  ChatMessage,
  ChatResponse,
  ObservabilitySummaryResponse,
  StatusResponse,
  VadSettings,
  VoiceInfo,
} from "./types";
import "./styles.css";

function uniqueValues(values: string[]): string[] {
  const seen = new Set<string>();
  return values.filter((value) => {
    if (!value || seen.has(value)) {
      return false;
    }
    seen.add(value);
    return true;
  });
}

function messageId(): string {
  return crypto.randomUUID?.() ?? `${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function observabilityRouteActive(): boolean {
  return /\/observability\/?$/.test(window.location.pathname);
}

function assistantMessage(response: ChatResponse): ChatMessage {
  return {
    id: messageId(),
    role: "assistant",
    content: response.tutor_response,
    modelName: response.model_name,
    voiceLabel: response.voice?.label,
    audioUrl: response.audio_url,
    ttsError: response.tts_error,
    sources: response.sources,
    retrievalCount: response.retrieval_count,
    retrievalError: response.retrieval_error,
  };
}

function userMessage(content: string, pronunciationFeedback?: string | null): ChatMessage {
  return {
    id: messageId(),
    role: "user",
    content,
    pronunciationFeedback,
  };
}

export default function App() {
  const showObservabilityPage = observabilityRouteActive();
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [selectedMode, setSelectedMode] = useState("free");
  const [selectedModel, setSelectedModel] = useState("");
  const [enableTts, setEnableTts] = useState(true);
  const [vadSettings, setVadSettings] = useState<VadSettings | null>(null);
  const [observabilitySummary, setObservabilitySummary] = useState<ObservabilitySummaryResponse | null>(null);
  const [observabilityLoading, setObservabilityLoading] = useState(true);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function refreshObservabilitySummary() {
    try {
      const summary = await fetchObservabilitySummary();
      setObservabilitySummary(summary);
    } catch {
      // Keep the existing dashboard state if the summary endpoint is temporarily unavailable.
    } finally {
      setObservabilityLoading(false);
    }
  }

  useEffect(() => {
    Promise.all([fetchStatus(), fetchObservabilitySummary()])
      .then(([statusPayload, summaryPayload]) => {
        setStatus(statusPayload);
        setObservabilitySummary(summaryPayload);
        setSelectedMode(statusPayload.default_mode);
        setSelectedModel(statusPayload.default_model);
        setEnableTts(statusPayload.tts_enabled);
        setVadSettings(statusPayload.vad);
        if (statusPayload.ollama_error) {
          setError(statusPayload.ollama_error);
        }
      })
      .catch((loadError: unknown) => {
        setError(loadError instanceof Error ? loadError.message : "Could not load API status.");
      })
      .finally(() => {
        setObservabilityLoading(false);
      });
  }, []);

  const modelOptions = useMemo(() => {
    if (!status) {
      return [];
    }
    return uniqueValues([status.default_model, ...status.recommended_models, ...status.installed_models]);
  }, [status]);

  const selectedVoice: VoiceInfo | null = status?.model_voices[selectedModel] ?? null;
  const currentMode = status?.modes.find((mode) => mode.key === selectedMode);

  async function refreshFocusWords(words: string[]) {
    setStatus((current) => (current ? { ...current, focus_words: words } : current));
  }

  async function handleAddFocusWord(word: string) {
    const response = await addFocusWord(word);
    await refreshFocusWords(response.words);
  }

  async function handleRemoveFocusWord(word: string) {
    const response = await removeFocusWord(word);
    await refreshFocusWords(response.words);
  }

  async function handleReset() {
    setBusy(true);
    setError(null);
    try {
      await resetSession(sessionId);
      setSessionId(null);
      setMessages([]);
      await refreshObservabilitySummary();
    } catch (resetError) {
      setError(resetError instanceof Error ? resetError.message : "Could not reset the session.");
    } finally {
      setBusy(false);
    }
  }

  async function runExchange(exchange: () => Promise<ChatResponse>) {
    setBusy(true);
    setError(null);
    try {
      const response = await exchange();
      setSessionId(response.session_id);
      setMessages((current) => [...current, assistantMessage(response)]);
      await refreshObservabilitySummary();
    } catch (exchangeError) {
      setError(exchangeError instanceof Error ? exchangeError.message : "The tutor request failed.");
    } finally {
      setBusy(false);
    }
  }

  async function handleSendText(message: string) {
    setMessages((current) => [...current, userMessage(message)]);
    await runExchange(() =>
      sendChat({
        message,
        sessionId,
        mode: selectedMode,
        modelName: selectedModel,
        enableTts,
      }),
    );
  }

  async function handleSendAudio(audio: Blob) {
    await runExchange(async () => {
      const response = await sendVoice({
        audio,
        sessionId,
        mode: selectedMode,
        modelName: selectedModel,
        enableTts,
      });
      setMessages((current) => [
        ...current,
        userMessage(response.user_text, response.pronunciation_feedback),
      ]);
      return response;
    });
  }

  if (!status) {
    return (
      <main className="loading-screen">
        <div className="pending-bubble">
          <span className="typing-dot" />
          <span className="typing-dot" />
          <span className="typing-dot" />
        </div>
      </main>
    );
  }

  if (showObservabilityPage) {
    return (
      <ObservabilityPage
        loading={observabilityLoading}
        onRefresh={refreshObservabilitySummary}
        summary={observabilitySummary}
      />
    );
  }

  return (
    <div className="app-shell">
      <Sidebar
        assistantName={status.assistant_name}
        busy={busy}
        contextTurns={status.context_turns}
        enableTts={enableTts}
        focusWords={status.focus_words}
        focusWordsLimit={status.focus_words_limit}
        installedModels={status.installed_models}
        modes={status.modes}
        models={modelOptions}
        observabilityLoading={observabilityLoading}
        observabilitySummary={observabilitySummary}
        onAddFocusWord={handleAddFocusWord}
        onEnableTtsChange={setEnableTts}
        onModeChange={setSelectedMode}
        onModelChange={setSelectedModel}
        onRemoveFocusWord={handleRemoveFocusWord}
        onReset={handleReset}
        onVadChange={setVadSettings}
        selectedMode={selectedMode}
        selectedModel={selectedModel}
        selectedVoice={selectedVoice}
        userDisplayName={status.user_display_name}
        vad={vadSettings ?? status.vad}
      />
      <main className="conversation-area">
        <ChatWindow
          error={error}
          messages={messages}
          mode={currentMode}
          modelName={selectedModel}
          pending={busy}
          voiceLabel={enableTts ? selectedVoice?.label ?? null : null}
        />
        <VoiceDock busy={busy} onSendAudio={handleSendAudio} onSendText={handleSendText} vad={vadSettings ?? status.vad} />
      </main>
    </div>
  );
}
