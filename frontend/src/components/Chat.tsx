import clsx from "clsx";
import { ArrowUp, Loader2, RotateCcw, Sparkles } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import type { AskResponse, Example, Mode } from "../api";
import { MODE_HINTS, MODE_LABELS } from "../theme";
import AnswerCard from "./AnswerCard";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  response?: AskResponse;
  error?: string;
  gold?: string;
}

const THINKING = [
  "Analyzing the question…",
  "Finding seed entities…",
  "Spreading activation through the graph…",
  "Collecting paths…",
  "Generating a grounded answer…",
  "Validating evidence…",
];

interface Props {
  messages: Message[];
  examples: Example[];
  loading: boolean;
  mode: Mode;
  selectedId: string | null;
  onModeChange: (mode: Mode) => void;
  onAsk: (question: string, gold?: string) => void;
  onSelect: (id: string) => void;
  onReset: () => void;
}

export default function Chat({
  messages,
  examples,
  loading,
  mode,
  selectedId,
  onModeChange,
  onAsk,
  onSelect,
  onReset,
}: Props) {
  const [draft, setDraft] = useState("");
  const [thinkingIndex, setThinkingIndex] = useState(0);
  const scrollRef = useRef<HTMLDivElement>(null);

  // scroll only the message list (scrollIntoView would also shift the page layout)
  useEffect(() => {
    const list = scrollRef.current;
    if (list) list.scrollTo({ top: list.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    if (!loading) return;
    setThinkingIndex(0);
    const timer = setInterval(
      () => setThinkingIndex((i) => Math.min(i + 1, THINKING.length - 1)),
      1400,
    );
    return () => clearInterval(timer);
  }, [loading]);

  const submit = () => {
    const question = draft.trim();
    if (!question || loading) return;
    setDraft("");
    onAsk(question);
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div ref={scrollRef} className="scroll-thin min-h-0 flex-1 space-y-4 overflow-y-auto pb-4 pr-1">
        {messages.length === 0 && (
          <div className="pt-6">
            <h1 className="font-display text-3xl font-bold leading-tight md:text-4xl">
              Ask the graph.
              <br />
              <span className="text-gradient">See how it thinks.</span>
            </h1>
            <p className="mt-4 max-w-lg text-sm leading-relaxed text-ink-300">
              Multi-hop questions over a knowledge graph extracted from documents. Retrieval
              runs <b className="text-ink-100">query-aware spreading activation</b> inside
              Neo4j, and every answer comes with the subgraph and paths it was built from.
            </p>
            <div className="mt-6 text-[11px] uppercase tracking-wider text-ink-500">
              Try one of these
            </div>
            <div className="mt-2 flex flex-wrap gap-2">
              {examples.map((example) => (
                <button
                  key={example.question}
                  onClick={() => onAsk(example.question, example.answer || undefined)}
                  disabled={loading}
                  className="rounded-2xl border border-white/10 bg-white/[0.03] px-3 py-2 text-left text-[13px] text-ink-300 transition hover:border-brand-violet/40 hover:bg-brand-violet/10 hover:text-ink-100"
                >
                  {example.question}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((message) =>
          message.role === "user" ? (
            <div key={message.id} className="flex justify-end">
              <div className="max-w-[85%] rounded-2xl rounded-br-md bg-gradient-to-br from-brand-violet/30 to-brand-cyan/15 px-4 py-2.5 text-sm text-ink-100">
                {message.content}
              </div>
            </div>
          ) : message.response ? (
            <AnswerCard
              key={message.id}
              response={message.response}
              gold={message.gold}
              selected={selectedId === message.id}
              onSelect={() => onSelect(message.id)}
            />
          ) : (
            <div
              key={message.id}
              className="rounded-2xl border border-status-critical/30 bg-status-critical/10 p-4 text-sm text-ink-100"
            >
              {message.error ?? "Something went wrong"}
            </div>
          ),
        )}

        {loading && (
          <div className="glass overflow-hidden rounded-2xl p-4">
            <div className="flex items-center gap-2 text-sm text-ink-300">
              <Loader2 className="h-4 w-4 animate-spin text-brand-cyan" />
              {THINKING[thinkingIndex]}
            </div>
            <div className="shimmer mt-3 h-2 rounded-full" />
            <div className="shimmer mt-2 h-2 w-2/3 rounded-full" />
          </div>
        )}
      </div>

      <div className="glass rounded-3xl p-2">
        <div className="flex items-center gap-1 px-1 pb-2 pt-1">
          {(["qasa", "uniform", "naive"] as Mode[]).map((m) => (
            <button
              key={m}
              onClick={() => onModeChange(m)}
              title={MODE_HINTS[m]}
              className={clsx(
                "rounded-full px-3 py-1 text-xs font-medium transition",
                mode === m
                  ? "bg-gradient-to-r from-brand-violet to-brand-cyan text-ink-950"
                  : "text-ink-300 hover:bg-white/5",
              )}
            >
              {MODE_LABELS[m]}
            </button>
          ))}
          <span className="ml-2 hidden truncate text-[11px] text-ink-500 md:inline">
            {MODE_HINTS[mode]}
          </span>
          {messages.length > 0 && (
            <button
              onClick={onReset}
              className="ml-auto flex items-center gap-1 rounded-full px-2.5 py-1 text-[11px] text-ink-500 transition hover:bg-white/5 hover:text-ink-300"
              title="Clear the conversation (and its memory)"
            >
              <RotateCcw className="h-3 w-3" /> new chat
            </button>
          )}
        </div>
        <div className="flex items-end gap-2">
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            rows={1}
            placeholder="Ask a multi-hop question, follow-ups work too…"
            className="scroll-thin max-h-32 min-h-[44px] flex-1 resize-none bg-transparent px-3 py-2.5 text-sm text-ink-100 placeholder:text-ink-500 focus:outline-none"
          />
          <button
            onClick={submit}
            disabled={loading || !draft.trim()}
            className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-brand-violet to-brand-cyan text-ink-950 transition hover:brightness-110 disabled:opacity-30"
          >
            {loading ? <Sparkles className="h-5 w-5 animate-pulse" /> : <ArrowUp className="h-5 w-5" />}
          </button>
        </div>
      </div>
    </div>
  );
}
