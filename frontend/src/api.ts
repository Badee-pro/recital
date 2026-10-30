// Types mirror the JSON returned by app.py.

export interface Contract { id: string; title: string; clauses: number; open_flags: number }
export interface Meta { methods: string[]; model: string; embed_model: string; contracts: Contract[]; examples: string[] }

export interface Clause { n: number; id: string; doc_id: string; doc_title: string; section: string; clause: string; text: string }
export interface SearchResult { ms: number; clauses: Clause[] }

export interface Grounding { top_similarity: number; threshold: number; gated: boolean; uncited: string[] }
export interface AnswerResult { ms: number; model: string; answer: string; cited: string[]; grounding: Grounding }

export interface Doc { id: string; title: string; markdown: string }

export type FlagStatus = "open" | "confirmed" | "dismissed";
export type Severity = "high" | "medium" | "low";
export interface Flag {
  id: string; doc_id: string; doc_title: string; check: string; check_title: string;
  clause_id: string; clause: string; section: string; severity: Severity; reason: string; model: string;
  status: FlagStatus; reviewer: string; note: string; created_at: string; reviewed_at: string | null;
}

export type JobState = "queued" | "running" | "done" | "error";
export interface Job { state: JobState; done: number; total: number; error?: string }
export type Jobs = Record<string, Job>;

type Scores = Record<string, number>;
export interface Stats { n_questions: number; retrieval: Record<string, Scores>; generation: Record<string, Scores> }

export async function api<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error((data as { detail?: string }).detail || `Request failed (${res.status})`);
  return data as T;
}
