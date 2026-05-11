import { useEffect, useRef } from "react";
import { absoluteAudioUrl } from "../api";
import type { ChatMessage, ModeInfo } from "../types";
import { FormattedMessage } from "./FormattedMessage";

interface ChatWindowProps {
  messages: ChatMessage[];
  pending: boolean;
  error: string | null;
  mode: ModeInfo | undefined;
  modelName: string;
  voiceLabel: string | null;
}

export function ChatWindow({
  messages,
  pending,
  error,
  mode,
  modelName,
  voiceLabel,
}: ChatWindowProps) {
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, pending, error]);

  return (
    <section className="chat-scroll" aria-live="polite">
      <div className="conversation-status">
        <div>
          <strong>{mode?.label ?? "Free Conversation"}</strong>
          <span>{modelName}</span>
        </div>
        {voiceLabel && <span>{voiceLabel}</span>}
      </div>

      {messages.length === 0 && (
        <div className="empty-state">
          <h2>Ready when you are.</h2>
        </div>
      )}

      <div className="message-list">
        {messages.map((message) => (
          <article className={`message-row ${message.role}`} key={message.id}>
            <div className="message-bubble">
              <div className="message-meta">
                <span>{message.role === "user" ? "You" : "Tutor"}</span>
                {message.modelName && <span>{message.modelName}</span>}
                {message.voiceLabel && <span>{message.voiceLabel}</span>}
              </div>
              <FormattedMessage content={message.content} />
              {message.pronunciationFeedback && (
                <div className="feedback-note">{message.pronunciationFeedback}</div>
              )}
              {message.ttsError && <div className="feedback-note warning">{message.ttsError}</div>}
              {message.audioUrl && (
                <audio className="answer-audio" controls autoPlay src={absoluteAudioUrl(message.audioUrl)} />
              )}
            </div>
          </article>
        ))}
        {pending && (
          <article className="message-row assistant">
            <div className="message-bubble pending-bubble">
              <span className="typing-dot" />
              <span className="typing-dot" />
              <span className="typing-dot" />
            </div>
          </article>
        )}
        {error && <div className="app-error">{error}</div>}
      </div>
      <div ref={endRef} />
    </section>
  );
}
