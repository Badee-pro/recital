import { useState } from "react";
import type { Contract, Flag, FlagStatus } from "../api";
import { pairOf, partiesOf, refOf } from "../format";
import { isActive, type Review } from "../useReview";
import type { OpenDocFn } from "../App";

const SEVERITY_ORDER = { high: 0, medium: 1, low: 2 };
const TABS: FlagStatus[] = ["open", "confirmed", "dismissed"];

function stored(key: string): string {
  try { return localStorage.getItem(`recital.${key}`) || ""; } catch { return ""; }
}
function store(key: string, value: string) {
  try { localStorage.setItem(`recital.${key}`, value); } catch { /* private mode: not remembered */ }
}

interface Props { review: Review; contracts: Contract[]; onOpenDoc: OpenDocFn }

export default function ReviewQueue({ review, contracts, onOpenDoc }: Props) {
  const [filter, setFilter] = useState<FlagStatus>("open");
  const [reviewer, setReviewer] = useState(() => stored("reviewer"));

  const titleOf = (docId: string) => contracts.find((c) => c.id === docId)?.title || docId;
  const shown = review.flags
    .filter((f) => f.status === filter)
    .sort((a, b) => SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity] || a.doc_id.localeCompare(b.doc_id));
  const jobs = Object.entries(review.jobs);
  const active = jobs.filter(([, j]) => isActive(j));
  const failed = jobs.filter(([, j]) => j.state === "error");

  return (
    <div className="queue">
      <div className="col">
        <div className="queue-head">
          <h1>Review queue</h1>
          <p>The AI checks each agreement for risky terms and flags the clause. It only proposes; a lawyer confirms or dismisses each flag.</p>
          <div className="tabs caps" role="group" aria-label="Filter flags">
            {TABS.map((s) => (
              <button key={s} className={filter === s ? "on" : ""} onClick={() => setFilter(s)}>{s}</button>
            ))}
          </div>
          <label className="field caps">Reviewer{" "}
            <input
              value={reviewer} placeholder="your name" autoComplete="name"
              onChange={(e) => { setReviewer(e.target.value); store("reviewer", e.target.value); }}
            />
          </label>
          <button className="ghost caps" onClick={() => review.runReview(contracts.map((c) => c.id))}>Review all agreements</button>
          <div className="jobs" role="status">
            {active.map(([id, j]) => `${pairOf(titleOf(id))}: ${j.state === "queued" ? "queued" : `check ${j.done + 1} of ${j.total}`}`).join(" · ")}
            {failed.map(([id]) => <span key={id} className="err"> {pairOf(titleOf(id))}: review failed. Is Ollama running?</span>)}
          </div>
        </div>

        <div>
          {shown.length ? shown.map((f) => (
            <FlagCard key={f.id} flag={f} reviewer={reviewer} decide={review.decide} onOpenDoc={onOpenDoc} />
          )) : (
            <div className="queue-empty">{filter === "open" ? "Nothing waiting for review." : `No ${filter} flags yet.`}</div>
          )}
        </div>
      </div>
    </div>
  );
}

interface CardProps { flag: Flag; reviewer: string; decide: Review["decide"]; onOpenDoc: OpenDocFn }

function FlagCard({ flag: f, reviewer, decide, onOpenDoc }: CardProps) {
  const [note, setNote] = useState(f.note || "");
  const [saving, setSaving] = useState(false);

  async function act(status: FlagStatus) {
    setSaving(true);
    try { await decide(f.id, status, reviewer, note); }
    catch (err) { alert((err as Error).message); }
    finally { setSaving(false); }
  }

  const when = f.status !== "open"
    ? `${f.status === "confirmed" ? "Confirmed" : "Dismissed"}${f.reviewer ? ` by ${f.reviewer}` : ""} · ${new Date(f.reviewed_at!).toLocaleString()}`
    : `Flagged by ${f.model} · ${new Date(f.created_at).toLocaleString()}`;

  return (
    <article className={`flag sev-${f.severity}`}>
      <div className="flag-top">
        <span className="sev">{f.severity}</span>
        <span className="check">{f.check_title}</span>
        <button
          className="ref" title={`Open ${partiesOf(f.doc_title)}`}
          onClick={() => onOpenDoc(f.doc_id, [f.clause_id], [], f.clause, "Flagged for review")}
        >
          {refOf(f)} · {pairOf(f.doc_title)}
        </button>
      </div>
      <p className="why">{f.reason || "No reason given."}</p>
      <span className="when">{when}</span>
      <textarea
        className="note" rows={1} placeholder="lawyer's note (optional)" aria-label="Note"
        value={note} onChange={(e) => setNote(e.target.value)}
      />
      <div className="acts">
        {f.status === "open" ? (
          <>
            <button className="yes" disabled={saving} onClick={() => act("confirmed")}>Confirm risk</button>
            <button className="no" disabled={saving} onClick={() => act("dismissed")}>Dismiss</button>
          </>
        ) : (
          <button className="re" disabled={saving} onClick={() => act("open")}>Reopen</button>
        )}
      </div>
    </article>
  );
}
