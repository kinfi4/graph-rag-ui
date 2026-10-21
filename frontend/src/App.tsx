import { useCallback, useEffect, useMemo, useState } from "react";

import { api, type Example, type Health, type Mode, type Stats } from "./api";
import Chat, { type Message } from "./components/Chat";
import DocsDrawer from "./components/DocsDrawer";
import GraphPanel from "./components/GraphPanel";
import Header from "./components/Header";

const newId = () => Math.random().toString(36).slice(2, 10);

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState<Mode>("qasa");
  const [health, setHealth] = useState<Health | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [examples, setExamples] = useState<Example[]>([]);
  const [docsOpen, setDocsOpen] = useState(false);

  const refreshStats = useCallback(() => {
    api.stats().then(setStats).catch(() => setStats(null));
  }, []);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
    api.examples().then(setExamples).catch(() => setExamples([]));
    refreshStats();
  }, [refreshStats]);

  const ask = async (question: string, gold?: string) => {
    // Conversation memory: the backend's analyzer resolves follow-ups from these turns.
    const history = messages
      .filter((m) => m.role === "user" || m.response)
      .slice(-6)
      .map((m) => ({
        role: m.role,
        content:
          m.role === "user"
            ? m.content
            : [m.response?.answer, m.response?.explanation].filter(Boolean).join(". "),
      }));

    setMessages((prev) => [...prev, { id: newId(), role: "user", content: question }]);
    setLoading(true);
    const id = newId();
    try {
      const response = await api.ask({ question, history, mode });
      setMessages((prev) => [
        ...prev,
        { id, role: "assistant", content: response.answer, response, gold },
      ]);
      if (response.graph.nodes.length > 0) setSelectedId(id);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        { id, role: "assistant", content: "", error: e instanceof Error ? e.message : String(e) },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const selected = useMemo(
    () => messages.find((m) => m.id === selectedId)?.response ?? null,
    [messages, selectedId],
  );

  return (
    <div className="relative flex h-full flex-col overflow-y-auto lg:overflow-hidden">
      {/* decorative glow, clipped so it never adds scrollable height */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute -left-40 -top-40 h-[28rem] w-[28rem] rounded-full bg-brand-violet/15 blur-[120px]" />
        <div className="absolute -bottom-48 right-0 h-[30rem] w-[30rem] rounded-full bg-brand-cyan/10 blur-[120px]" />
      </div>

      <Header health={health} stats={stats} onOpenDocs={() => setDocsOpen(true)} />

      {/* the row must be minmax(0,1fr): an auto row grows with the chat and shifts the page */}
      <main className="relative grid min-h-0 flex-1 grid-cols-1 gap-5 px-6 pb-6 lg:grid-cols-[minmax(360px,5fr)_7fr] lg:grid-rows-[minmax(0,1fr)]">
        <section className="h-[80vh] min-h-0 lg:h-auto">
          <Chat
            messages={messages}
            examples={examples}
            loading={loading}
            mode={mode}
            selectedId={selectedId}
            onModeChange={setMode}
            onAsk={ask}
            onSelect={setSelectedId}
            onReset={() => {
              setMessages([]);
              setSelectedId(null);
            }}
          />
        </section>
        <section className="min-h-[520px] lg:min-h-0">
          <GraphPanel response={selected} />
        </section>
      </main>

      <DocsDrawer open={docsOpen} onClose={() => setDocsOpen(false)} onIngested={refreshStats} />
    </div>
  );
}
