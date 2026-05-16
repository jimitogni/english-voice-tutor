import { ArrowLeft, RefreshCw } from "lucide-react";
import type { ObservabilitySummaryResponse } from "../types";
import { ObservabilityPanel } from "./ObservabilityPanel";

interface ObservabilityPageProps {
  summary: ObservabilitySummaryResponse | null;
  loading: boolean;
  onRefresh: () => Promise<void>;
}

function appHref(): string {
  const pathname = window.location.pathname;
  return pathname.replace(/\/observability\/?$/, "/");
}

export function ObservabilityPage({ summary, loading, onRefresh }: ObservabilityPageProps) {
  return (
    <main className="observability-page">
      <div className="observability-page-shell">
        <div className="observability-page-header">
          <a className="observability-nav-link" href={appHref()}>
            <ArrowLeft size={16} aria-hidden="true" />
            Back to tutor
          </a>
          <button className="observability-refresh-button" onClick={() => void onRefresh()} type="button">
            <RefreshCw size={16} aria-hidden="true" />
            Refresh
          </button>
        </div>

        <section className="observability-hero">
          <span className="observability-kicker">Production view</span>
          <h1>Observability</h1>
          <p>
            Follow interaction volume, latency, errors, tool behavior, feedback, and the latest
            evaluation run from one page.
          </p>
        </section>

        <ObservabilityPanel loading={loading} summary={summary} />
      </div>
    </main>
  );
}
