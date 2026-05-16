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

export interface RagStatus {
  enabled: boolean;
  vector_db: string;
  embedding_model: string;
  collection: string;
  top_k: number;
  score_threshold: number;
  qdrant_url: string;
  knowledge_dir: string;
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
  stt_device: string;
  stt_compute_type: string;
  piper_cuda: boolean;
  vad: VadSettings;
  rag: RagStatus;
}

export interface ObservabilityRunSummary {
  run_id: string;
  created_at: string;
  dataset_path: string;
  dataset_size: number;
  model_name: string | null;
  git_commit: string | null;
  records_path: string;
  summary_path: string;
  averages: Record<string, number>;
  counts: Record<string, number>;
}

export interface ObservabilitySummaryResponse {
  status: string;
  service: string;
  environment: string;
  langfuse_enabled: boolean;
  langfuse_url: string | null;
  evaluation_enabled: boolean;
  metrics_enabled: boolean;
  prometheus_enabled: boolean;
  rag_enabled: boolean;
  total_interactions: number;
  interactions_last_24h: number;
  total_errors: number;
  average_latency_ms: number;
  average_feedback_score: number | null;
  tool_call_count: number;
  tool_call_error_count: number;
  task_success_rate: number | null;
  last_interaction_at: string | null;
  latest_run: ObservabilityRunSummary | null;
}

export interface RagSourceInfo {
  title: string;
  source: string;
  score: number | null;
  content_preview: string;
}

export interface ChatResponse {
  request_id: string;
  session_id: string;
  mode: string;
  model_name: string;
  voice: VoiceInfo | null;
  user_text: string;
  tutor_response: string;
  audio_url: string | null;
  tts_error: string | null;
  pronunciation_feedback: string | null;
  sources: RagSourceInfo[];
  retrieval_count: number;
  retrieval_error: string | null;
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
  sources?: RagSourceInfo[];
  retrievalCount?: number;
  retrievalError?: string | null;
}
