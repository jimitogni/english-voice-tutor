import { Activity, BarChart3, CircleAlert, Gauge, Radar, Wrench } from "lucide-react";
import type { ObservabilitySummaryResponse } from "../types";

interface ObservabilityPanelProps {
  summary: ObservabilitySummaryResponse | null;
  loading: boolean;
}

function formatTimestamp(value: string | null): string {
  if (!value) {
    return "No data yet";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString();
}

function formatSuccessRate(value: number | null): string {
  if (value === null) {
    return "n/a";
  }
  return `${Math.round(value * 100)}%`;
}

function formatFeedback(value: number | null): string {
  if (value === null) {
    return "n/a";
  }
  return `${value.toFixed(1)}/5`;
}

export function ObservabilityPanel({ summary, loading }: ObservabilityPanelProps) {
  if (loading) {
    return (
      <section className="panel-section">
        <div className="section-heading">Observability</div>
        <div className="observability-card">
          <p className="observability-muted">Loading summary...</p>
        </div>
      </section>
    );
  }

  if (!summary) {
    return (
      <section className="panel-section">
        <div className="section-heading">Observability</div>
        <div className="observability-card">
          <p className="observability-muted">Summary unavailable.</p>
        </div>
      </section>
    );
  }

  return (
    <section className="panel-section">
      <div className="section-heading">Observability</div>
      <div className="observability-card">
        <div className="observability-badges">
          <span className={summary.evaluation_enabled ? "status-badge on" : "status-badge off"}>
            <Radar size={14} aria-hidden="true" />
            Eval
          </span>
          <span className={summary.prometheus_enabled ? "status-badge on" : "status-badge off"}>
            <Activity size={14} aria-hidden="true" />
            Metrics
          </span>
          <span className={summary.langfuse_enabled ? "status-badge on" : "status-badge off"}>
            <BarChart3 size={14} aria-hidden="true" />
            Traces
          </span>
        </div>

        {summary.langfuse_enabled && summary.langfuse_url && (
          <p className="observability-muted">
            <a href={summary.langfuse_url} rel="noreferrer" target="_blank">
              Open Langfuse traces
            </a>
          </p>
        )}

        <div className="observability-grid">
          <div className="metric-tile">
            <span>Interactions</span>
            <strong>{summary.total_interactions}</strong>
            <small>{summary.interactions_last_24h} in 24h</small>
          </div>
          <div className="metric-tile">
            <span>Errors</span>
            <strong>{summary.total_errors}</strong>
            <small>{formatSuccessRate(summary.task_success_rate)} success</small>
          </div>
          <div className="metric-tile">
            <span>Latency</span>
            <strong>{summary.average_latency_ms.toFixed(0)} ms</strong>
            <small>avg response</small>
          </div>
          <div className="metric-tile">
            <span>Feedback</span>
            <strong>{formatFeedback(summary.average_feedback_score)}</strong>
            <small>user score</small>
          </div>
        </div>

        <div className="observability-stats">
          <div>
            <Wrench size={15} aria-hidden="true" />
            <span>Tool calls</span>
            <strong>{summary.tool_call_count}</strong>
          </div>
          <div>
            <CircleAlert size={15} aria-hidden="true" />
            <span>Tool errors</span>
            <strong>{summary.tool_call_error_count}</strong>
          </div>
          <div>
            <Gauge size={15} aria-hidden="true" />
            <span>Last interaction</span>
            <strong>{formatTimestamp(summary.last_interaction_at)}</strong>
          </div>
        </div>

        {summary.latest_run && (
          <div className="observability-run">
            <div className="observability-run-header">
              <strong>Latest Eval Run</strong>
              <span>{new Date(summary.latest_run.created_at).toLocaleString()}</span>
            </div>
            <p>
              Dataset: <code>{summary.latest_run.dataset_path}</code>
            </p>
            <p>
              Records: {summary.latest_run.dataset_size} · Model: {summary.latest_run.model_name ?? "unknown"}
            </p>
            <div className="observability-run-metrics">
              <span>Semantic {summary.latest_run.averages.semantic_similarity?.toFixed(2) ?? "0.00"}</span>
              <span>BLEU {summary.latest_run.averages.bleu_score?.toFixed(2) ?? "0.00"}</span>
              <span>Latency {summary.latest_run.averages.latency_ms?.toFixed(0) ?? "0"} ms</span>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
