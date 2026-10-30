import { useEffect, useRef } from "react";
import type { Stats } from "../api";
import { methodLabel, pct } from "../format";

interface Props { open: boolean; stats: Stats | null; onClose: () => void }

export default function AboutDialog({ open, stats, onClose }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const d = dialog.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);

  const r = stats?.retrieval ?? {};
  const g = stats?.generation.local;
  const best = Object.entries(r).sort((a, b) => b[1].mrr - a[1].mrr)[0]?.[0];

  return (
    <dialog
      ref={dialog}
      onClose={onClose}
      onClick={(e) => { if (e.target === dialog.current) onClose(); }}
    >
      <div className="dlg-head">
        <h2>About Recital</h2>
        <button className="x" aria-label="Close" onClick={onClose}>×</button>
      </div>
      <div className="dlg-body">
        <p>Recital is a retrieval-augmented question-answering system for contracts, built by <a href="https://badeekhalbouss.com" target="_blank" rel="noopener">Badee Khalbouss</a>. It answers from the agreements only, cites the clause behind every claim, and declines when the agreements don't cover a question.</p>
        <h4>How an answer is made</h4>
        <ol>
          <li><b>Split by clause.</b> Each agreement is cut at its numbered clauses. Every clause keeps its agreement and section name.</li>
          <li><b>Retrieve.</b> The question is embedded with bge-small and compared with all clauses. Keyword (BM25) and hybrid search are available to compare.</li>
          <li><b>Answer.</b> A local qwen2.5:7b, served by Ollama, writes an answer from the retrieved clauses only and tags each sentence with its source. Tags that don't match a retrieved clause are discarded.</li>
          <li><b>Check grounding in code.</b> If no clause is close in meaning to the question (similarity below 0.60), the model is never asked and the question is declined. Any answer sentence without a clause citation is underlined and flagged.</li>
          <li><b>Measure.</b> Every change is scored on hand-labelled questions and logged to MLflow.</li>
        </ol>
        <h4>Retrieval, by search method</h4>
        <table>
          <tbody>
            <tr><th>Method</th><th>Top 1</th><th>Top 3</th><th>Top 5</th><th>MRR</th></tr>
            {Object.entries(r).map(([m, v]) => (
              <tr key={m} className={m === best ? "best" : ""}>
                <td>{methodLabel(m)}</td><td>{pct(v.recall_at_1)}</td><td>{pct(v.recall_at_3)}</td><td>{pct(v.recall_at_5)}</td><td>{v.mrr.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <h4>Answers · local model, embedding search, top 5</h4>
        <table>
          <tbody>
            {g ? (
              <>
                <tr><th>Measure</th><th>Score</th></tr>
                <tr><td>Key fact appears in the answer</td><td>{pct(g.answer_match)}</td></tr>
                <tr><td>Cites the clause that holds the answer</td><td>{pct(g.citation_precision)}</td></tr>
                <tr><td>Declines when the agreements don't cover it</td><td>{pct(g.abstain_correct)}</td></tr>
              </>
            ) : <tr><td>Run the evaluation to see answer scores.</td></tr>}
          </tbody>
        </table>
        <p style={{ marginTop: 16 }}>Keyword search scores lowest because questions paraphrase ("if they get hacked") while contracts use defined terms ("Personal Data Breach"). Hybrid search inherits some of that noise, so embedding search is the default.</p>
      </div>
    </dialog>
  );
}
