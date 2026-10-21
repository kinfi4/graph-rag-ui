import clsx from "clsx";
import { FileText, GitBranch, ListTree, Pause, Play, Sparkles, Waypoints } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import type { AskResponse, GraphNode } from "../api";
import { MODE_LABELS, TYPE_COLORS, TYPE_LABELS, typeColor } from "../theme";
import GraphCanvas from "./GraphCanvas";

type Tab = "paths" | "sources" | "trace";

export default function GraphPanel({ response }: { response: AskResponse | null }) {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [tab, setTab] = useState<Tab>("paths");
  const [hoveredPath, setHoveredPath] = useState<string | null>(null);
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);

  const steps = response?.steps ?? 0;
  const nodes = response?.graph.nodes ?? [];
  const links = response?.graph.links ?? [];

  // Every new answer replays the spreading activation from the seeds.
  useEffect(() => {
    setSelectedNode(null);
    setHoveredPath(null);
    if (response && response.steps > 0) {
      setStep(0);
      setPlaying(true);
    } else {
      setStep(response?.steps ?? 0);
      setPlaying(false);
    }
  }, [response]);

  useEffect(() => {
    if (!playing) return;
    if (step >= steps) {
      setPlaying(false);
      return;
    }
    const timer = setTimeout(() => setStep((s) => s + 1), 900);
    return () => clearTimeout(timer);
  }, [playing, step, steps]);

  const highlight = useMemo(() => {
    if (!hoveredPath || !response) return null;
    const path = response.paths.find((p) => p.id === hoveredPath);
    return path ? new Set(path.names) : null;
  }, [hoveredPath, response]);

  const activeCount = nodes.filter(
    (n) => (n.activation[Math.min(step, n.activation.length - 1)] ?? 0) > 0,
  ).length;
  const typesPresent = Object.keys(TYPE_COLORS).filter((t) => nodes.some((n) => n.type === t));
  const inspected = hoveredNode ?? selectedNode;

  if (!response || nodes.length === 0) {
    return (
      <div className="glass relative flex h-full flex-col items-center justify-center overflow-hidden rounded-3xl p-10 text-center">
        <div className="pointer-events-none absolute -top-20 left-1/4 h-64 w-64 rounded-full bg-brand-violet/20 blur-3xl animate-float" />
        <div className="pointer-events-none absolute bottom-0 right-10 h-72 w-72 rounded-full bg-brand-cyan/10 blur-3xl animate-float [animation-delay:-3s]" />
        <Waypoints className="mb-5 h-12 w-12 text-brand-violet" strokeWidth={1.4} />
        <h2 className="font-display text-xl font-semibold">The reasoning graph shows up here</h2>
        <p className="mt-3 max-w-md text-sm leading-relaxed text-ink-300">
          Ask a question, and you'll see where the search starts (seeds), how activation
          spreads through the knowledge graph step by step, and which paths the model
          used for its answer.
        </p>
      </div>
    );
  }

  return (
    <div className="glass flex h-full min-h-0 flex-col overflow-hidden rounded-3xl">
      {/* header */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-white/5 px-5 py-3">
        <div className="flex items-center gap-2">
          <GitBranch className="h-4 w-4 text-brand-violet" />
          <span className="font-display text-sm font-semibold tracking-wide">Reasoning graph</span>
          <span className="rounded-full bg-white/5 px-2 py-0.5 font-mono text-[11px] text-ink-300">
            {MODE_LABELS[response.mode] ?? response.mode}
          </span>
        </div>
        <div className="ml-auto flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-ink-300">
          {typesPresent.map((type) => (
            <span key={type} className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-full" style={{ background: typeColor(type) }} />
              {TYPE_LABELS[type]}
            </span>
          ))}
          <span className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full border border-dashed border-ink-100" />
            seed
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-white shadow-[0_0_8px_white]" />
            evidence
          </span>
        </div>
      </div>

      {/* canvas */}
      <div className="relative min-h-[260px] flex-1 bg-[radial-gradient(ellipse_at_center,rgba(167,139,250,0.07),transparent_65%)]">
        <GraphCanvas
          nodes={nodes}
          links={links}
          step={step}
          highlight={highlight}
          selected={selectedNode?.id ?? null}
          onHover={setHoveredNode}
          onSelect={setSelectedNode}
        />
        {inspected && <Inspector node={inspected} step={step} />}
        <div className="pointer-events-none absolute right-4 top-3 text-right font-mono text-[11px] text-ink-500">
          {nodes.length} nodes · {links.length} edges
        </div>
      </div>

      {/* step slider */}
      {steps > 0 && (
        <div className="flex items-center gap-3 border-t border-white/5 px-5 py-3">
          <button
            onClick={() => {
              if (step >= steps) setStep(0);
              setPlaying((p) => !p);
            }}
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-brand-violet/20 text-brand-violet transition hover:bg-brand-violet/30"
            title="Replay spreading activation"
          >
            {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          </button>
          <input
            type="range"
            min={0}
            max={steps}
            value={step}
            onChange={(e) => {
              setPlaying(false);
              setStep(Number(e.target.value));
            }}
            className="step-slider flex-1"
          />
          <div className="w-44 shrink-0 text-right font-mono text-xs text-ink-300">
            step {step}/{steps} ·{" "}
            <span className="text-ink-100">
              {step === 0 ? "seeds" : `${activeCount} active`}
            </span>
          </div>
        </div>
      )}

      {/* details tabs */}
      <div className="flex h-[38%] min-h-[180px] flex-col border-t border-white/5">
        <div className="flex gap-1 px-4 pt-3">
          <TabButton icon={Waypoints} active={tab === "paths"} onClick={() => setTab("paths")}>
            Paths <span className="text-ink-500">{response.paths.length}</span>
          </TabButton>
          <TabButton icon={FileText} active={tab === "sources"} onClick={() => setTab("sources")}>
            Sources <span className="text-ink-500">{response.sources.length}</span>
          </TabButton>
          <TabButton icon={ListTree} active={tab === "trace"} onClick={() => setTab("trace")}>
            Pipeline trace
          </TabButton>
        </div>
        <div className="scroll-thin min-h-0 flex-1 overflow-y-auto px-4 pb-4 pt-3">
          {tab === "paths" && (
            <PathsTab response={response} onHover={setHoveredPath} hovered={hoveredPath} />
          )}
          {tab === "sources" && <SourcesTab response={response} />}
          {tab === "trace" && <TraceTab response={response} />}
        </div>
      </div>
    </div>
  );
}

function TabButton({
  icon: Icon,
  active,
  onClick,
  children,
}: {
  icon: typeof Waypoints;
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={clsx(
        "flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium transition",
        active ? "bg-white/10 text-ink-100" : "text-ink-300 hover:bg-white/5",
      )}
    >
      <Icon className="h-3.5 w-3.5" />
      {children}
    </button>
  );
}

function Inspector({ node, step }: { node: GraphNode; step: number }) {
  const peak = Math.max(1e-6, ...node.activation);
  return (
    <div className="glass absolute bottom-3 left-3 w-80 max-w-[calc(100%-1.5rem)] rounded-2xl p-4 text-xs shadow-2xl">
      <div className="flex items-center gap-2">
        <span className="h-2.5 w-2.5 rounded-full" style={{ background: typeColor(node.type) }} />
        <span className="text-[10px] uppercase tracking-wider text-ink-300">
          {TYPE_LABELS[node.type] ?? node.type}
        </span>
        {node.is_seed && (
          <span className="rounded-full bg-white/10 px-2 py-0.5 text-[10px]">seed</span>
        )}
        {node.evidence && (
          <span className="rounded-full bg-brand-lime/15 px-2 py-0.5 text-[10px] text-brand-lime">
            evidence
          </span>
        )}
      </div>
      <div className="mt-1.5 text-sm font-semibold capitalize text-ink-100">{node.id}</div>
      {node.description && (
        <p className="mt-1.5 line-clamp-4 leading-relaxed text-ink-300">{node.description}</p>
      )}
      {node.activation.length > 1 && (
        <div className="mt-3">
          <div className="mb-1 text-[10px] uppercase tracking-wider text-ink-500">
            activation by step
          </div>
          <div className="flex h-8 items-end gap-1">
            {node.activation.map((value, i) => (
              <div key={i} className="flex flex-1 flex-col items-center gap-0.5">
                <div
                  className={clsx(
                    "w-full rounded-t-[3px]",
                    i === step ? "bg-brand-violet" : "bg-brand-violet/35",
                  )}
                  style={{ height: `${Math.max(2, (value / peak) * 26)}px` }}
                  title={value.toFixed(3)}
                />
              </div>
            ))}
          </div>
          <div className="mt-0.5 flex gap-1 font-mono text-[9px] text-ink-500">
            {node.activation.map((value, i) => (
              <span key={i} className="flex-1 text-center">
                {value.toFixed(2)}
              </span>
            ))}
          </div>
        </div>
      )}
      {node.documents.length > 0 && (
        <div className="mt-2 truncate text-[10px] text-ink-500">
          from: {node.documents.join(" · ")}
        </div>
      )}
    </div>
  );
}

function PathsTab({
  response,
  hovered,
  onHover,
}: {
  response: AskResponse;
  hovered: string | null;
  onHover: (id: string | null) => void;
}) {
  if (response.paths.length === 0) {
    return (
      <p className="text-sm text-ink-300">
        {response.mode === "naive"
          ? "Vector mode retrieves entities only, without traversing relations."
          : "No paths in this answer."}
      </p>
    );
  }
  const evidence = new Set(response.evidence);
  const sorted = [...response.paths].sort(
    (a, b) => Number(evidence.has(b.id)) - Number(evidence.has(a.id)) || b.weight - a.weight,
  );
  const maxWeight = Math.max(...response.paths.map((p) => p.weight), 1e-6);

  return (
    <ul className="space-y-1.5">
      {sorted.map((path) => {
        const isEvidence = evidence.has(path.id);
        const parts = path.text.split(/( --\[.*?\]--> | <--\[.*?\]-- )/);
        return (
          <li
            key={path.id}
            onMouseEnter={() => onHover(path.id)}
            onMouseLeave={() => onHover(null)}
            className={clsx(
              "flex cursor-default items-start gap-3 rounded-xl px-3 py-2 transition",
              isEvidence ? "bg-brand-lime/[0.07] ring-1 ring-brand-lime/25" : "bg-white/[0.03]",
              hovered === path.id && "bg-white/[0.08]",
            )}
          >
            <span
              className={clsx(
                "mt-0.5 shrink-0 rounded-md px-1.5 py-0.5 font-mono text-[10px]",
                isEvidence ? "bg-brand-lime/20 text-brand-lime" : "bg-white/10 text-ink-300",
              )}
            >
              {path.id}
            </span>
            <div className="min-w-0 flex-1 text-[13px] leading-relaxed">
              {parts.map((part, i) =>
                i % 2 === 1 ? (
                  <span key={i} className="mx-1 font-mono text-[11px] text-brand-violet">
                    {part.trim()}
                  </span>
                ) : (
                  <span key={i} className="capitalize text-ink-100">
                    {part}
                  </span>
                ),
              )}
            </div>
            <div className="mt-2 h-1 w-14 shrink-0 overflow-hidden rounded-full bg-white/5">
              <div
                className={isEvidence ? "h-full bg-brand-lime" : "h-full bg-brand-violet/60"}
                style={{ width: `${(path.weight / maxWeight) * 100}%` }}
                title={`path weight ${path.weight}`}
              />
            </div>
          </li>
        );
      })}
    </ul>
  );
}

function highlightHits(text: string, hits: string[]) {
  if (hits.length === 0) return text;
  const escaped = hits
    .filter((h) => h.length > 2)
    .sort((a, b) => b.length - a.length)
    .map((h) => h.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  if (escaped.length === 0) return text;
  const pattern = new RegExp(`(${escaped.join("|")})`, "gi");
  return text.split(pattern).map((part, i) =>
    i % 2 === 1 ? (
      <mark key={i} className="rounded bg-brand-violet/25 px-0.5 text-ink-100">
        {part}
      </mark>
    ) : (
      part
    ),
  );
}

function SourcesTab({ response }: { response: AskResponse }) {
  if (response.sources.length === 0) {
    return <p className="text-sm text-ink-300">No source paragraphs for this answer.</p>;
  }
  return (
    <div className="space-y-2">
      <p className="text-[11px] text-ink-500">
        Provenance: original paragraphs that mention the evidence entities (the model saw
        only the graph, these show where the graph came from).
      </p>
      {response.sources.map((source, i) => (
        <div key={i} className="rounded-xl bg-white/[0.03] p-3">
          <div className="mb-1 flex items-center gap-2 text-xs font-semibold text-ink-100">
            <FileText className="h-3.5 w-3.5 text-brand-cyan" />
            {source.title}
            <span className="font-normal text-ink-500">· {source.hits.length} entities</span>
          </div>
          <p className="text-[13px] leading-relaxed text-ink-300">
            {highlightHits(source.text, source.hits)}
          </p>
        </div>
      ))}
    </div>
  );
}

function TraceTab({ response }: { response: AskResponse }) {
  const timings = Object.entries(response.timings).filter(([k]) => k !== "total");
  const total = response.timings.total || 1;
  return (
    <div className="grid gap-4 text-xs md:grid-cols-2">
      <div className="space-y-3">
        <TraceRow label="1 · Analyzer + router">
          route <b className="text-ink-100">{response.route}</b> · language{" "}
          <b className="text-ink-100">{response.language}</b>
          <div className="mt-1 italic text-ink-300">“{response.standalone_question}”</div>
        </TraceRow>
        <TraceRow label="2 · Query entities → seeds">
          <div className="flex flex-wrap gap-1">
            {response.query_entities.map((e) => (
              <span key={e.name} className="rounded-full bg-white/5 px-2 py-0.5">
                {e.name}
              </span>
            ))}
            {response.seeds.length > 0 && <span className="text-ink-500">→</span>}
            {response.seeds.map((s) => (
              <span key={s} className="rounded-full border border-dashed border-ink-300/60 px-2 py-0.5">
                {s}
              </span>
            ))}
          </div>
        </TraceRow>
        {response.attempts.length > 0 && (
          <TraceRow label="3 · Retrieval → generation → validation">
            <table className="w-full font-mono text-[11px]">
              <thead className="text-ink-500">
                <tr>
                  <td>try</td>
                  <td>hops</td>
                  <td>paths</td>
                  <td>answer</td>
                  <td>suff.</td>
                  <td>grounded</td>
                </tr>
              </thead>
              <tbody>
                {response.attempts.map((a, i) => (
                  <tr key={i} className="text-ink-300">
                    <td>#{i + 1}</td>
                    <td>{a.max_steps}</td>
                    <td>{a.paths}</td>
                    <td className="max-w-[9rem] truncate text-ink-100">{a.answer || "—"}</td>
                    <td>{a.sufficient ? "✓" : "✗"}</td>
                    <td>{a.grounded ? "✓" : "✗"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {response.refined && (
              <div className="mt-1 flex items-center gap-1 text-brand-lime">
                <Sparkles className="h-3 w-3" /> refined: second pass with a wider traversal
              </div>
            )}
          </TraceRow>
        )}
      </div>
      <div className="space-y-3">
        {response.reasoning && (
          <TraceRow label="Model reasoning">
            <p className="leading-relaxed text-ink-300">{response.reasoning}</p>
          </TraceRow>
        )}
        <TraceRow label={`Latency · ${total.toFixed(1)} s · ${response.llm}`}>
          <div className="flex h-2 w-full overflow-hidden rounded-full bg-white/5">
            {timings.map(([key, value], i) => (
              <div
                key={key}
                className={["bg-brand-violet", "bg-brand-cyan", "bg-brand-lime"][i % 3]}
                style={{ width: `${(value / total) * 100}%` }}
              />
            ))}
          </div>
          <div className="mt-1.5 flex gap-3 font-mono text-[11px] text-ink-300">
            {timings.map(([key, value], i) => (
              <span key={key} className="flex items-center gap-1">
                <span
                  className={clsx(
                    "h-2 w-2 rounded-full",
                    ["bg-brand-violet", "bg-brand-cyan", "bg-brand-lime"][i % 3],
                  )}
                />
                {key} {value.toFixed(2)}s
              </span>
            ))}
          </div>
        </TraceRow>
      </div>
    </div>
  );
}

function TraceRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl bg-white/[0.03] p-3">
      <div className="mb-1.5 text-[10px] uppercase tracking-wider text-ink-500">{label}</div>
      <div className="text-ink-300">{children}</div>
    </div>
  );
}
