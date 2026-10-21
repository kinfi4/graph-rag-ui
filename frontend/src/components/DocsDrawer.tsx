import clsx from "clsx";
import { CheckCircle2, FileUp, Loader2, Search, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { api, type DocumentRow, type Job } from "../api";

export default function DocsDrawer({
  open,
  onClose,
  onIngested,
}: {
  open: boolean;
  onClose: () => void;
  onIngested: () => void;
}) {
  const [documents, setDocuments] = useState<DocumentRow[]>([]);
  const [filter, setFilter] = useState("");
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const refresh = () => api.documents().then(setDocuments).catch(() => setDocuments([]));

  useEffect(() => {
    if (open) refresh();
  }, [open]);

  useEffect(() => {
    if (!job || job.stage === "done" || job.stage === "error") return;
    const timer = setTimeout(async () => {
      try {
        const next = await api.job(job.id);
        setJob(next);
        if (next.stage === "done") {
          refresh();
          onIngested();
        }
      } catch (e) {
        setError(String(e));
      }
    }, 1200);
    return () => clearTimeout(timer);
  }, [job, onIngested]);

  const upload = async (files: File[]) => {
    if (files.length === 0) return;
    setError(null);
    try {
      const { job_id } = await api.upload(files);
      setJob({ id: job_id, stage: "queued", done: 0, total: 0, message: "" });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const busy = job !== null && job.stage !== "done" && job.stage !== "error";
  const visible = documents.filter((d) => d.title.toLowerCase().includes(filter.toLowerCase()));

  return (
    <>
      <div
        onClick={onClose}
        className={clsx(
          "fixed inset-0 z-40 bg-black/50 backdrop-blur-sm transition",
          open ? "opacity-100" : "pointer-events-none opacity-0",
        )}
      />
      <aside
        className={clsx(
          "fixed right-0 top-0 z-50 flex h-full w-full max-w-md flex-col border-l border-white/10 bg-ink-900 transition-transform duration-300",
          open ? "translate-x-0" : "translate-x-full",
        )}
      >
        <div className="flex items-center justify-between px-5 py-4">
          <div className="font-display font-semibold">Documents</div>
          <button onClick={onClose} className="rounded-full p-1.5 hover:bg-white/5">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="px-5">
          <div
            onClick={() => !busy && inputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              if (!busy) upload(Array.from(e.dataTransfer.files));
            }}
            className={clsx(
              "flex cursor-pointer flex-col items-center rounded-2xl border border-dashed px-4 py-6 text-center transition",
              dragging ? "border-brand-cyan bg-brand-cyan/10" : "border-white/15 hover:border-brand-violet/50",
              busy && "cursor-wait opacity-60",
            )}
          >
            <FileUp className="h-6 w-6 text-brand-violet" />
            <div className="mt-2 text-sm text-ink-100">Drop .txt / .md / .pdf files</div>
            <div className="mt-1 text-[11px] text-ink-500">
              chunks → LLM extraction → validation → Neo4j → embeddings
            </div>
            <input
              ref={inputRef}
              type="file"
              multiple
              accept=".txt,.md,.pdf"
              className="hidden"
              onChange={(e) => upload(Array.from(e.target.files ?? []))}
            />
          </div>

          {job && (
            <div className="mt-3 rounded-xl bg-white/[0.04] p-3 text-xs">
              <div className="flex items-center gap-2 text-ink-100">
                {busy ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin text-brand-cyan" />
                ) : (
                  <CheckCircle2 className="h-3.5 w-3.5 text-status-good" />
                )}
                <span className="capitalize">{job.stage}</span>
                {job.total > 0 && (
                  <span className="text-ink-500">
                    {job.done}/{job.total} chunks
                  </span>
                )}
              </div>
              {job.total > 0 && (
                <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-white/5">
                  <div
                    className="h-full bg-gradient-to-r from-brand-violet to-brand-cyan transition-all"
                    style={{ width: `${(job.done / job.total) * 100}%` }}
                  />
                </div>
              )}
              {job.message && <div className="mt-1.5 truncate text-ink-500">{job.message}</div>}
              {job.report && (
                <div className="mt-2 grid grid-cols-2 gap-1 font-mono text-[11px] text-ink-300">
                  <span>entities: {String(job.report.entities)}</span>
                  <span>relations: {String(job.report.triples)}</span>
                  <span>dropped: {String(job.report.dropped_entities)}</span>
                  <span>repaired: {String(job.report.repaired_endpoints)}</span>
                  <span>failed chunks: {String(job.report.failed_chunks)}</span>
                  <span>{String(job.report.seconds)} s</span>
                </div>
              )}
            </div>
          )}
          {error && <div className="mt-3 text-xs text-status-critical">{error}</div>}

          <div className="mt-5 flex items-center gap-2 rounded-xl bg-white/[0.04] px-3 py-2">
            <Search className="h-3.5 w-3.5 text-ink-500" />
            <input
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder={`Filter ${documents.length} documents`}
              className="flex-1 bg-transparent text-sm placeholder:text-ink-500 focus:outline-none"
            />
          </div>
        </div>

        <ul className="scroll-thin mt-3 min-h-0 flex-1 space-y-1 overflow-y-auto px-5 pb-5">
          {visible.map((doc) => (
            <li
              key={doc.title}
              className="flex items-center justify-between gap-3 rounded-lg px-3 py-2 text-sm hover:bg-white/[0.04]"
            >
              <span className="truncate text-ink-100">{doc.title}</span>
              <span className="flex shrink-0 items-center gap-2 text-[11px] text-ink-500">
                {doc.mentions} ent.
                <span
                  className={clsx(
                    "rounded-full px-2 py-0.5",
                    doc.source === "upload" ? "bg-brand-cyan/15 text-brand-cyan" : "bg-white/5",
                  )}
                >
                  {doc.source ?? "—"}
                </span>
              </span>
            </li>
          ))}
        </ul>
      </aside>
    </>
  );
}
