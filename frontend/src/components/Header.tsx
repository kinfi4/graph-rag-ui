import clsx from "clsx";
import { Database, FolderOpen } from "lucide-react";

import type { Health, Stats } from "../api";

const formatCount = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n));

export default function Header({
  health,
  stats,
  onOpenDocs,
}: {
  health: Health | null;
  stats: Stats | null;
  onOpenDocs: () => void;
}) {
  return (
    <header className="flex flex-wrap items-center gap-x-6 gap-y-3 px-6 py-4">
      <div className="flex items-center gap-3">
        <div className="relative h-9 w-9">
          <span className="absolute left-1/2 top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-brand-violet shadow-[0_0_16px_#a78bfa]" />
          <span className="absolute left-0 top-1 h-2 w-2 rounded-full bg-brand-cyan" />
          <span className="absolute right-0 top-2 h-2 w-2 rounded-full bg-brand-lime" />
          <span className="absolute bottom-0 right-2 h-2 w-2 rounded-full bg-brand-pink" />
        </div>
        <div>
          <div className="font-display text-lg font-bold leading-none tracking-tight">synapse</div>
          <div className="mt-1 text-[11px] text-ink-500">
            explainable GraphRAG · query-aware spreading activation
          </div>
        </div>
      </div>

      <div className="ml-auto flex flex-wrap items-center gap-2 text-xs">
        {stats && (
          <div className="glass flex items-center gap-3 rounded-full px-4 py-1.5 text-ink-300">
            <Database className="h-3.5 w-3.5 text-brand-cyan" />
            <span>
              <b className="text-ink-100">{formatCount(stats.documents)}</b> docs
            </span>
            <span>
              <b className="text-ink-100">{formatCount(stats.entities)}</b> entities
            </span>
            <span>
              <b className="text-ink-100">{formatCount(stats.relations)}</b> relations
            </span>
          </div>
        )}
        <div className="glass flex items-center gap-3 rounded-full px-4 py-1.5 text-ink-300">
          <StatusDot ok={health?.neo4j} label="Neo4j" />
          <StatusDot ok={health?.ollama} label={health?.llm ?? "LLM"} />
        </div>
        <button
          onClick={onOpenDocs}
          className="glass flex items-center gap-2 rounded-full px-4 py-1.5 text-ink-100 transition hover:border-brand-violet/40"
        >
          <FolderOpen className="h-3.5 w-3.5 text-brand-violet" />
          Documents
        </button>
      </div>
    </header>
  );
}

function StatusDot({ ok, label }: { ok: boolean | undefined; label: string }) {
  return (
    <span className="flex items-center gap-1.5" title={ok ? "online" : "offline"}>
      <span
        className={clsx(
          "h-2 w-2 rounded-full",
          ok === undefined ? "bg-ink-500" : ok ? "bg-status-good shadow-[0_0_8px_#0ca30c]" : "bg-status-critical",
        )}
      />
      <span className="font-mono">{label}</span>
    </span>
  );
}
