import clsx from "clsx";
import {
  AlertTriangle,
  CheckCircle2,
  CircleSlash,
  Globe,
  Info,
  MessageCircle,
  Search,
  Sparkles,
  Target,
} from "lucide-react";

import type { AskResponse } from "../api";
import { MODE_LABELS } from "../theme";

const STATUS = {
  answered: { icon: CheckCircle2, label: "Grounded in graph", tone: "text-status-good bg-status-good/10" },
  unverified: { icon: AlertTriangle, label: "Not verified", tone: "text-status-warning bg-status-warning/10" },
  not_found: { icon: CircleSlash, label: "Not in graph", tone: "text-status-critical bg-status-critical/10" },
  info: { icon: Info, label: "Info", tone: "text-ink-300 bg-white/5" },
} as const;

const ROUTES = {
  graph_qa: { icon: Search, label: "Graph QA" },
  overview: { icon: Globe, label: "Overview" },
  smalltalk: { icon: MessageCircle, label: "Small talk" },
} as const;

const normalize = (text: string) =>
  text
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[̀-ͯ]/g, "")
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .replace(/\b(a|an|the)\b/g, " ")
    .replace(/\s+/g, " ")
    .trim();

export default function AnswerCard({
  response,
  gold,
  selected,
  onSelect,
}: {
  response: AskResponse;
  gold?: string;
  selected: boolean;
  onSelect: () => void;
}) {
  const status = STATUS[response.status];
  const route = ROUTES[response.route];
  const hasGraph = response.graph.nodes.length > 0;
  const goldMatch =
    gold !== undefined && gold !== ""
      ? normalize(response.answer).includes(normalize(gold)) ||
        (normalize(response.answer).length > 0 && normalize(gold).includes(normalize(response.answer)))
      : null;

  return (
    <div
      onClick={hasGraph ? onSelect : undefined}
      className={clsx(
        "glass rounded-2xl p-4 transition",
        hasGraph && "cursor-pointer hover:border-white/15",
        selected && hasGraph && "border-brand-violet/50 shadow-[0_0_0_1px_rgba(167,139,250,0.35),0_10px_40px_-12px_rgba(167,139,250,0.35)]",
      )}
    >
      <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
        <span className="flex items-center gap-1 rounded-full bg-white/5 px-2 py-0.5 text-ink-300">
          <route.icon className="h-3 w-3" />
          {route.label}
        </span>
        {response.route === "graph_qa" && (
          <span className="rounded-full bg-brand-violet/15 px-2 py-0.5 font-mono text-brand-violet">
            {MODE_LABELS[response.mode] ?? response.mode}
          </span>
        )}
        <span className={clsx("flex items-center gap-1 rounded-full px-2 py-0.5", status.tone)}>
          <status.icon className="h-3 w-3" />
          {status.label}
        </span>
        {response.refined && (
          <span className="flex items-center gap-1 rounded-full bg-brand-lime/10 px-2 py-0.5 text-brand-lime">
            <Sparkles className="h-3 w-3" />
            refined
          </span>
        )}
      </div>

      {response.answer && (
        <div
          className={clsx(
            "mt-3 font-display font-semibold leading-snug",
            // graph keys are lowercase; capitalize names for display
            response.route === "graph_qa"
              ? "text-gradient text-xl capitalize"
              : "text-[15px] text-ink-100",
          )}
        >
          {response.answer}
        </div>
      )}
      {response.status === "not_found" && !response.answer && (
        <div className="mt-3 font-display text-lg font-semibold text-ink-300">
          The graph has no answer to this
        </div>
      )}
      {response.explanation && (
        <p className="mt-2 text-sm leading-relaxed text-ink-300">{response.explanation}</p>
      )}
      {response.highlights && response.highlights.length > 0 && (
        <ul className="mt-2 space-y-1 text-sm text-ink-300">
          {response.highlights.map((item) => (
            <li key={item} className="flex gap-2">
              <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-brand-cyan" />
              {item}
            </li>
          ))}
        </ul>
      )}

      {goldMatch !== null && (
        <div
          className={clsx(
            "mt-3 flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs",
            goldMatch ? "bg-status-good/10 text-status-good" : "bg-status-critical/10 text-status-critical",
          )}
        >
          <Target className="h-3.5 w-3.5" />
          2Wiki gold answer: <b>{gold}</b> {goldMatch ? "· match" : "· mismatch"}
        </div>
      )}

      <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[11px] text-ink-500">
        {response.route === "graph_qa" && (
          <span>
            {response.seeds.length} seeds · {response.paths.length} paths ·{" "}
            {response.evidence.length} evidence
          </span>
        )}
        <span>{response.timings.total?.toFixed(1)} s</span>
        {hasGraph && !selected && <span className="text-brand-violet">click to show graph →</span>}
      </div>
    </div>
  );
}
