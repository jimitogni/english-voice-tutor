import type { ChatResponse, FocusWordsResponse, StatusResponse } from "./types";

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...init?.headers,
    },
  });

  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = (await response.json()) as { detail?: unknown };
      if (typeof payload.detail === "string") {
        message = payload.detail;
      }
    } catch {
      const text = await response.text();
      if (text) {
        message = text;
      }
    }
    throw new Error(message);
  }

  return (await response.json()) as T;
}

export function absoluteAudioUrl(audioUrl: string): string {
  if (audioUrl.startsWith("http://") || audioUrl.startsWith("https://")) {
    return audioUrl;
  }
  if (!API_BASE_URL) {
    return audioUrl;
  }
  return `${API_BASE_URL}${audioUrl}`;
}

export function fetchStatus(): Promise<StatusResponse> {
  return requestJson<StatusResponse>("/api/status");
}

export interface SendChatInput {
  message: string;
  sessionId: string | null;
  mode: string;
  modelName: string;
  enableTts: boolean;
}

export function sendChat(input: SendChatInput): Promise<ChatResponse> {
  return requestJson<ChatResponse>("/api/chat", {
    method: "POST",
    body: JSON.stringify({
      message: input.message,
      session_id: input.sessionId,
      mode: input.mode,
      model_name: input.modelName,
      enable_tts: input.enableTts,
    }),
  });
}

export interface SendVoiceInput {
  audio: Blob;
  sessionId: string | null;
  mode: string;
  modelName: string;
  enableTts: boolean;
}

export function sendVoice(input: SendVoiceInput): Promise<ChatResponse> {
  const formData = new FormData();
  formData.append("file", input.audio, "browser-recording.webm");
  if (input.sessionId) {
    formData.append("session_id", input.sessionId);
  }
  formData.append("mode", input.mode);
  formData.append("model_name", input.modelName);
  formData.append("enable_tts", String(input.enableTts));

  return requestJson<ChatResponse>("/api/voice", {
    method: "POST",
    body: formData,
  });
}

export function addFocusWord(text: string): Promise<FocusWordsResponse> {
  return requestJson<FocusWordsResponse>("/api/focus-words", {
    method: "POST",
    body: JSON.stringify({ text }),
  });
}

export function removeFocusWord(word: string): Promise<FocusWordsResponse> {
  return requestJson<FocusWordsResponse>(`/api/focus-words/${encodeURIComponent(word)}`, {
    method: "DELETE",
  });
}

export function resetSession(sessionId: string | null): Promise<{ session_id: string; reset: boolean }> {
  return requestJson<{ session_id: string; reset: boolean }>("/api/reset", {
    method: "POST",
    body: JSON.stringify({ session_id: sessionId }),
  });
}
