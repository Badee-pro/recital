import { useRef, useState } from "react";
import { api, type AnswerResult, type Meta, type SearchResult } from "../api";
import { methodLabel } from "../format";
import type { OpenDocFn } from "../App";
import Turn, { type TurnData } from "./Turn";

interface Props { meta: Meta; onOpenDoc: OpenDocFn }

export default function Thread({ meta, onOpenDoc }: Props) {
  const [turns, setTurns] = useState<TurnData[]>([]);
  const [question, setQuestion] = useState("");
  const [method, setMethod] = useState("dense");
  const [k, setK] = useState(5);
  const [busy, setBusy] = useState(false);
  const input = useRef<HTMLTextAreaElement>(null);

  const update = (id: number, patch: Partial<TurnData>) =>
    setTurns((ts) => ts.map((t) => (t.id === id ? { ...t, ...patch } : t)));

  async function ask(text = question) {
    const q = text.trim();
    if (q.length < 3 || busy) return;
    setBusy(true);
    setQuestion("");
    const body = { question: q, method, k };
    const id = Date.now();
    setTurns((ts) => [...ts, { id, question: q, method, k, phase: "searching" }]);
    try {
      const found = await api<SearchResult>("/api/search", body);
      update(id, { phase: "answering", found, started: Date.now() });
      const ans = await api<AnswerResult>("/api/answer", body);
      const firstCited = found.clauses.find((c) => ans.cited.includes(c.id));
      update(id, { phase: "done", ans, selected: firstCited?.n });
      if (firstCited) {
        onOpenDoc(firstCited.doc_id, ans.cited, found.clauses.map((c) => c.id), firstCited.clause);
      }
    } catch (err) {
      update(id, { phase: "error", error: (err as Error).message });
    } finally {
      setBusy(false);
      input.current?.focus();
    }
  }

  function autosize(el: HTMLTextAreaElement) {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
  }

  return (
    <>
      <div className="thread">
        <div className="col">
          {turns.length === 0 ? (
            <div className="empty">
              <h1>Ask the agreements</h1>
              <div className="sub caps">Contract Q&amp;A · by <a href="https://badeekhalbouss.com" target="_blank" rel="noopener">Badee Khalbouss</a></div>
              <p>Answers come only from the four agreements. Every claim cites its clause, and the clause opens beside the answer. If the agreements don't say, Recital says so.</p>
              <div className="places caps">NDA · MSA · DPA · SaaS</div>
              <div className="suggest">
                {meta.examples.map((ex) => (
                  <button type="button" key={ex} onClick={() => ask(ex)}><span>{ex}</span><span>↵</span></button>
                ))}
              </div>
            </div>
          ) : (
            turns.map((t) => (
              <Turn
                key={t.id}
                turn={t}
                onSelect={(n) => update(t.id, { selected: n })}
                onOpenDoc={onOpenDoc}
              />
            ))
          )}
        </div>
      </div>

      <div className="composer">
        <div className="col">
          <form className="box" onSubmit={(e) => { e.preventDefault(); ask(); }}>
            <textarea
              ref={input}
              rows={1}
              value={question}
              placeholder="ask a question about the agreements"
              aria-label="Question"
              onChange={(e) => { setQuestion(e.target.value); autosize(e.target); }}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); } }}
            />
            <button className="send" type="submit" disabled={busy}>Ask</button>
          </form>
          <div className="opts caps">
            <label>Search{" "}
              <select value={method} onChange={(e) => setMethod(e.target.value)}>
                {meta.methods.map((m) => <option key={m} value={m}>{methodLabel(m)}</option>)}
              </select>
            </label>
            <label>Clauses{" "}
              <select value={k} onChange={(e) => setK(+e.target.value)}>
                {[3, 5, 8].map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </label>
            <span className="hint">Enter ↵ to ask</span>
          </div>
        </div>
      </div>
    </>
  );
}
