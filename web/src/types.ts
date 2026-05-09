export interface ModeInfo {
  key: string;
  label: string;
  description: string;
}

export interface VoiceInfo {
  key: string;
  label: string;
  model_path: string;
  config_path: string;
  available: boolean;
}

export interface VadSettings {
  energy_threshold: number;
  silence_seconds: number;
  min_speech_seconds: number;
  max_seconds: number;
  chunk_ms: number;
  sample_rate: number;
}

export interface StatusResponse {
  assistant_name: string;
  user_display_name: string;
  default_model: string;
  recommended_models: string[];
  installed_models: string[];
  ollama_error: string | null;
  modes: ModeInfo[];
  model_voices: Record<string, VoiceInfo>;
  voice_profiles: VoiceInfo[];
  default_mode: string;
  context_turns: number;
  focus_words: string[];
  focus_words_limit: number;
  tts_enabled: boolean;
  llm_stream_enabled: boolean;
  vad: VadSettings;
}

export interface ChatResponse {
  session_id: string;
  mode: string;
  model_name: string;
  voice: VoiceInfo | null;
  user_text: string;
  tutor_response: string;
  audio_url: string | null;
  tts_error: string | null;
  pronunciation_feedback: string | null;
}

export interface FocusWordsResponse {
  words: string[];
  limit: number;
}

export type ChatRole = "user" | "assistant";

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  modelName?: string;
  voiceLabel?: string;
  audioUrl?: string | null;
  pronunciationFeedback?: string | null;
  ttsError?: string | null;
}
