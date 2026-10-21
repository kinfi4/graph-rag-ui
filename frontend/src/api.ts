export type Mode = "qasa" | "uniform" | "naive";
export type Route = "graph_qa" | "overview" | "smalltalk";
export type Status = "answered" | "unverified" | "not_found" | "info";

export interface GraphNode {
  id: string;
  type: string;
  description: string;
  documents: string[];
  degree: number;
  is_seed: boolean;
  evidence: boolean;
  activation: number[];
  // force-graph adds these at runtime
  x?: number;
  y?: number;
}

export interface GraphLink {
  source: string | GraphNode;
  target: string | GraphNode;
  relation: string;
  on_path: boolean;
  evidence: boolean;
}

export interface PathItem {
  id: string;
  text: string;
  names: string[];
  weight: number;
}

export interface Attempt {
  max_steps: number;
  threshold: number;
  paths: number;
  entities: number;
  answer: string;
  sufficient: boolean;
  grounded: boolean;
}

export interface SourceParagraph {
  title: string;
  text: string;
  hits: string[];
}

export interface AskResponse {
  question: string;
  standalone_question: string;
  route: Route;
  language: string;
  query_entities: { name: string; search_hint: string }[];
  llm: string;
  timings: Record<string, number>;
  mode: string;
  status: Status;
  answer: string;
  explanation: string;
  reasoning: string;
  highlights?: string[];
  evidence: string[];
  grounded: boolean;
  attempts: Attempt[];
  refined: boolean;
  seeds: string[];
  paths: PathItem[];
  steps: number;
  graph: { nodes: GraphNode[]; links: GraphLink[] };
  sources: SourceParagraph[];
}

export interface Health {
  neo4j: boolean;
  ollama: boolean;
  llm: string;
  embedding: string;
}

export interface Stats {
  documents: number;
  paragraphs: number;
  entities: number;
  relations: number;
  mentions: number;
}

export interface DocumentRow {
  title: string;
  source: string | null;
  mentions: number;
}

export interface Example {
  question: string;
  answer: string;
}

export interface Job {
  id: string;
  stage: string;
  done: number;
  total: number;
  message: string;
  report?: Record<string, number | string[]>;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail ?? detail;
    } catch {
      /* keep statusText */
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<Health>("/api/health"),
  stats: () => request<Stats>("/api/stats"),
  documents: () => request<DocumentRow[]>("/api/documents"),
  examples: () => request<Example[]>("/api/examples"),
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  ask: (body: {
    question: string;
    history: { role: "user" | "assistant"; content: string }[];
    mode: Mode;
  }) =>
    request<AskResponse>("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  upload: (files: File[]) => {
    const form = new FormData();
    files.forEach((file) => form.append("files", file));
    return request<{ job_id: string }>("/api/documents", { method: "POST", body: form });
  },
};
