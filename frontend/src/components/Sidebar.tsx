import { useRef, useState } from "react";
import { api, type Contract, type Jobs, type Stats } from "../api";
import { pairOf, pct, shortOf } from "../format";

interface Props {
  show: boolean;
  contracts: Contract[];
  jobs: Jobs;
  activeDocId?: string;
  stats: Stats | null;
  onOpen: (docId: string) => void;
  onUploaded: () => void;
  onAbout: () => void;
}

export default function Sidebar({ show, contracts, jobs, activeDocId, stats, onOpen, onUploaded, onAbout }: Props) {
  const fileInput = useRef<HTMLInputElement>(null);
  const [upload, setUpload] = useState<{ text: string; err?: boolean }>({ text: "" });

  async function onFile(file: File) {
    setUpload({ text: `uploading ${file.name}…` });
    try {
      const d = await api<{ id: string; clauses: number }>("/api/contracts", { filename: file.name, markdown: await file.text() });
      setUpload({ text: `added ${d.clauses} clauses · AI review queued` });
      onUploaded();
    } catch (err) {
      setUpload({ text: (err as Error).message, err: true });
    }
  }

  return (
    <aside className={`side${show ? " show" : ""}`}>
      <div>
        <h6 className="caps">Agreements · <span>{contracts.length}</span></h6>
        <div>
          {contracts.map((d) => (
            <button key={d.id} className={`doc-item${d.id === activeDocId ? " on" : ""}`} onClick={() => onOpen(d.id)}>
              <span className="kind">{shortOf(d.id) || "DOC"}</span>
              <span className="who">{pairOf(d.title)}</span>
              <span className="n">{d.clauses} clauses<DocStatus contract={d} jobs={jobs} /></span>
            </button>
          ))}
        </div>
        <button className="link" onClick={() => fileInput.current?.click()}>Upload agreement</button>
        <input
          ref={fileInput} type="file" hidden accept=".md,.txt,text/markdown,text/plain"
          onChange={(e) => { const f = e.target.files?.[0]; e.target.value = ""; if (f) onFile(f); }}
        />
        <div className={`upload-msg${upload.err ? " err" : ""}`} role="status">{upload.text}</div>
      </div>

      <div>
        <h6 className="caps">Evaluation</h6>
        <EvalStats stats={stats} />
        <button className="link" onClick={onAbout}>How it's measured</button>
      </div>

      <div className="foot">Fictional contracts for demonstration. Not legal advice.</div>
    </aside>
  );
}

function DocStatus({ contract, jobs }: { contract: Contract; jobs: Jobs }) {
  const job = jobs[contract.id];
  if (job?.state === "queued") return <> · <span className="badge">review queued</span></>;
  if (job?.state === "running") return <> · <span className="badge">reviewing {job.done + 1}/{job.total}</span></>;
  if (contract.open_flags) return <> · <span className="badge">{contract.open_flags} flag{contract.open_flags > 1 ? "s" : ""}</span></>;
  return null;
}

function EvalStats({ stats }: { stats: Stats | null }) {
  if (!stats) return <div className="stat"><span>Loading…</span></div>;
  const g = stats.generation.local;
  const rows: [string, number | undefined][] = [["Right clause, top 5", stats.retrieval.dense.recall_at_5]];
  if (g) rows.push(["Citations correct", g.citation_precision], ["Facts correct", g.answer_match], ["Correct refusals", g.abstain_correct]);
  return (
    <div>
      {rows.map(([k, v]) => <div className="stat" key={k}><span>{k}</span><b>{pct(v)}</b></div>)}
      <div className="stat"><span>Test questions</span><b>{stats.n_questions}</b></div>
    </div>
  );
}
