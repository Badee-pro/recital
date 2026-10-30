import { useEffect, useRef, useState } from "react";
import { api, type Doc } from "../api";
import { kindLabel, partiesOf } from "../format";

export interface OpenDoc {
  docId: string;
  hits: string[];   // clause ids highlighted gold (cited or flagged)
  seen: string[];   // clause ids that were retrieved but not cited
  focus?: string;   // clause number to scroll to, e.g. "7.2"
  note: string;     // label under a highlighted clause
  nonce: number;    // re-scroll even when the same clause is opened twice
}

type Block = { kind: "h"; text: string } | { kind: "p"; text: string } | { kind: "clause"; no: string; text: string };

const cache = new Map<string, Promise<Doc>>();
const loadDoc = (id: string) => {
  if (!cache.has(id)) cache.set(id, api<Doc>(`/api/contracts/${id}`));
  return cache.get(id)!;
};

function parse(markdown: string): Block[] {
  const blocks: Block[] = [];
  for (const block of markdown.split(/\n\s*\n/).map((b) => b.trim()).filter(Boolean)) {
    if (block.startsWith("# ")) continue;
    if (block.startsWith("## ")) { blocks.push({ kind: "h", text: block.slice(3) }); continue; }
    const m = block.match(/^(\d+\.\d+)\s+([\s\S]*)$/);
    blocks.push(m ? { kind: "clause", no: m[1], text: m[2] } : { kind: "p", text: block });
  }
  return blocks;
}

interface Props { doc: OpenDoc; onClose: () => void; onReview: (docId: string) => void }

export default function DocViewer({ doc, onClose, onReview }: Props) {
  const [loaded, setLoaded] = useState<Doc | null>(null);
  const body = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let live = true;
    loadDoc(doc.docId).then((d) => { if (live) setLoaded(d); }).catch(console.error);
    return () => { live = false; };
  }, [doc.docId]);

  // Wait for the pane to finish opening before scrolling, as the original page did.
  useEffect(() => {
    if (!loaded || loaded.id !== doc.docId) return;
    const t = window.setTimeout(() => {
      const target = doc.focus ? body.current?.querySelector(`[data-clause="${doc.focus}"]`) : null;
      if (target) target.scrollIntoView({ behavior: "smooth", block: "center" });
      else if (body.current) body.current.scrollTop = 0;
    }, 220);
    return () => window.clearTimeout(t);
  }, [loaded, doc.docId, doc.focus, doc.nonce]);

  const current = loaded?.id === doc.docId ? loaded : null;

  return (
    <section className="doc" aria-label="Agreement viewer">
      <div className="doc-head">
        <div className="t">
          <div className="k caps">{kindLabel(doc.docId)}</div>
          <h2>{current ? partiesOf(current.title) : ""}</h2>
        </div>
        <button className="ghost caps" onClick={() => onReview(doc.docId)}>Run AI review</button>
        <button className="x" aria-label="Close agreement" onClick={onClose}>×</button>
      </div>
      <div className="doc-body" ref={body}>
        {current && parse(current.markdown).map((b, i) => {
          if (b.kind === "h") return <h3 key={i}>{b.text}</h3>;
          if (b.kind === "p") return <p key={i}>{b.text}</p>;
          const id = `${doc.docId}#${b.no}`;
          const cls = doc.hits.includes(id) ? "hit" : doc.seen.includes(id) ? "seen" : "";
          return (
            <div key={i} className={`cl ${cls}`} data-clause={b.no}>
              <span className="no">{b.no}</span>
              <span>{b.text}</span>
              {cls === "hit" && <span className="note">{doc.note}</span>}
            </div>
          );
        })}
      </div>
    </section>
  );
}
