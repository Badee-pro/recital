import { useEffect, useRef, useState, type ReactNode } from "react";
import type { AnswerResult, Clause, SearchResult } from "../api";
import { methodLabel, refOf } from "../format";
import type { OpenDocFn } from "../App";

export interface TurnData {
  id: number;
  question: string;
  method: string;
  k: number;
  phase: "searching" | "answering" | "done" | "error";
  found?: SearchResult;
  ans?: AnswerResult;
  started?: number;
  selected?: number;  // which retrieved clause (1-based) is shown in the viewer
  error?: string;
}

interface Props { turn: TurnData; onSelect: (n: number) => void; onOpenDoc: OpenDocFn }

export default function Turn({ turn, onSelect, onOpenDoc }: Props) {
  const el = useRef<HTMLElement>(null);
  useEffect(() => { el.current?.scrollIntoView({ behavior: "smooth", block: "start" }); }, []);

  const { found, ans } = turn;
  if (turn.phase !== "done" || !found || !ans) {
    return (
      <article className="turn" ref={el}>
        <div className="q">{turn.question}</div>
        <Status turn={turn} />
      </article>
    );
  }

  const clauses = found.clauses;
  const show = (n: number) => {
    const c = clauses[n - 1];
    if (!c) return;
    onSelect(n);
    onOpenDoc(c.doc_id, ans.cited, clauses.map((x) => x.id), c.clause);
  };

  // A refusal is an uncited "don't address" answer; a cited answer that merely ends with that line isn't one.
  const none = !ans.cited.length && /don.t address/i.test(ans.answer);
  const g = ans.grounding;
  const cited = clauses.filter((c) => ans.cited.includes(c.id));
  const others = clauses.filter((c) => !ans.cited.includes(c.id));
  const row = (c: Clause, isCited: boolean) => (
    <li key={c.id} className={`${isCited ? "" : "uncited"}${turn.selected === c.n ? " on" : ""}`} onClick={() => show(c.n)}>
      <span className="i">{c.n}</span>
      <span className="ref">{refOf(c)}</span>
      <span className="sec">{c.section.replace(/^\d+\.\s*/, "")}</span>
      <span className="tag">{isCited ? "Cited" : ""}</span>
    </li>
  );

  return (
    <article className="turn" ref={el}>
      <div className="q">{turn.question}</div>
      <p className={`a${none ? " none" : ""}`}>{renderAnswer(ans.answer, g.uncited, clauses, show)}</p>
      {g.gated ? (
        <div className="ground">
          Declined before asking the model: no clause is close to this question (similarity {g.top_similarity.toFixed(2)}, needs {g.threshold.toFixed(2)}).
        </div>
      ) : g.uncited.length > 0 && (
        <div className="ground warn">
          {g.uncited.length} sentence{g.uncited.length > 1 ? "s cite" : " cites"} no clause (underlined). Check before relying on {g.uncited.length > 1 ? "them" : "it"}.
        </div>
      )}
      <ol className="srcs">{cited.map((c) => row(c, true))}{others.map((c) => row(c, false))}</ol>
      <div className="meta">
        {g.gated ? "model not called" : ans.model} · {(ans.ms / 1000).toFixed(1)}s · {methodLabel(turn.method).toLowerCase()} search {found.ms} ms
      </div>
    </article>
  );
}

function Status({ turn }: { turn: TurnData }) {
  const [, tick] = useState(0);
  useEffect(() => {
    if (turn.phase !== "answering") return;
    const t = window.setInterval(() => tick((n) => n + 1), 300);
    return () => window.clearInterval(t);
  }, [turn.phase]);

  if (turn.phase === "error") return <div className="status err">error: {turn.error}</div>;
  const label = turn.phase === "answering" && turn.found
    ? `found ${turn.found.clauses.length} clauses in ${turn.found.ms} ms · drafting answer ${Math.round((Date.now() - (turn.started ?? Date.now())) / 1000)}s`
    : `searching ${turn.k} most relevant clauses…`;
  return <div className="status"><span className="cursor" /><span>{label}</span></div>;
}

/** Answer text with "[n]" markers turned into citation buttons and uncited sentences underlined. */
function renderAnswer(text: string, uncited: string[], clauses: Clause[], show: (n: number) => void): ReactNode[] {
  const pieces: { text: string; uncited: boolean }[] = [{ text, uncited: false }];
  for (const s of uncited) {
    const i = pieces.findIndex((p) => !p.uncited && p.text.includes(s));
    if (i < 0) continue;
    const p = pieces[i];
    const at = p.text.indexOf(s);
    pieces.splice(i, 1,
      { text: p.text.slice(0, at), uncited: false },
      { text: s, uncited: true },
      { text: p.text.slice(at + s.length), uncited: false });
  }

  const withCites = (t: string, key: string): ReactNode[] =>
    t.split(/\s*\[(\d+)\]/).map((part, j) => {
      if (j % 2 === 0) return part;
      const n = +part;
      const c = clauses[n - 1];
      return c ? <button key={`${key}-${j}`} className="cite" title={refOf(c)} onClick={() => show(n)}>{n}</button> : null;
    });

  return pieces.map((p, i) => p.uncited
    ? <span key={i} className="nocite" title="This sentence cites no clause">{withCites(p.text, `u${i}`)}</span>
    : <span key={i}>{withCites(p.text, `t${i}`)}</span>);
}
