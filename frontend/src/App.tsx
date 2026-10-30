import { useCallback, useEffect, useState } from "react";
import { api, type Contract, type Meta, type Stats } from "./api";
import { useReview } from "./useReview";
import Sidebar from "./components/Sidebar";
import Thread from "./components/Thread";
import ReviewQueue from "./components/ReviewQueue";
import DocViewer, { type OpenDoc } from "./components/DocViewer";
import AboutDialog from "./components/AboutDialog";

export type OpenDocFn = (docId: string, hits: string[], seen: string[], focus?: string, note?: string) => void;

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [modelName, setModelName] = useState("Connecting");
  const [contracts, setContracts] = useState<Contract[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [doc, setDoc] = useState<OpenDoc | null>(null);
  const [queueOpen, setQueueOpen] = useState(false);
  const [sideOpen, setSideOpen] = useState(false);
  const [aboutOpen, setAboutOpen] = useState(false);

  const refreshContracts = useCallback(async () => {
    setContracts((await api<Meta>("/api/meta")).contracts);
  }, []);
  const review = useReview(refreshContracts);

  useEffect(() => {
    api<Meta>("/api/meta")
      .then((m) => { setMeta(m); setContracts(m.contracts); setModelName(`${m.model} · local`); })
      .catch((err) => { setModelName("Server not reachable"); console.error(err); });
    api<Stats>("/api/stats").then(setStats).catch(console.error);
  }, []);

  const openDoc: OpenDocFn = (docId, hits, seen, focus, note = "Cited in answer") =>
    setDoc({ docId, hits, seen, focus, note, nonce: Date.now() });

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape" && !aboutOpen) setDoc(null); };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [aboutOpen]);

  const toggleQueue = (show = !queueOpen) => {
    setQueueOpen(show);
    if (show) review.loadFlags();
  };

  return (
    <div className="app">
      <header className="top">
        <button className="ghost menu-btn caps" aria-label="Agreements" onClick={() => setSideOpen((s) => !s)}>Menu</button>
        <a className="logo" href="/">Recital<span>.</span></a>
        <a className="by caps" href="https://badeekhalbouss.com" target="_blank" rel="noopener">by Badee Khalbouss</a>
        <div className="spacer" />
        <div className="model caps" title="Answering model">
          <span className={`led${meta ? " on" : ""}`} /><span>{modelName}</span>
        </div>
        <button className={`ghost caps${queueOpen ? " on" : ""}`} aria-pressed={queueOpen} onClick={() => toggleQueue()}>
          Review queue <b>{review.openCount || ""}</b>
        </button>
        <button className="ghost caps" onClick={() => setAboutOpen(true)}>About</button>
      </header>

      <div className={`body${doc ? " doc-open" : ""}`}>
        <Sidebar
          show={sideOpen}
          contracts={contracts}
          jobs={review.jobs}
          activeDocId={doc?.docId}
          stats={stats}
          onOpen={(id) => { openDoc(id, [], []); setSideOpen(false); }}
          onUploaded={async () => { await refreshContracts(); review.pollJobs(); }}
          onAbout={() => setAboutOpen(true)}
        />

        <main className={`main${queueOpen ? " queue-open" : ""}`}>
          <ReviewQueue review={review} contracts={contracts} onOpenDoc={openDoc} />
          {meta && <Thread meta={meta} onOpenDoc={openDoc} />}
        </main>

        {doc && (
          <DocViewer
            doc={doc}
            onClose={() => setDoc(null)}
            onReview={(id) => { review.runReview(id); toggleQueue(true); }}
          />
        )}
      </div>

      <AboutDialog open={aboutOpen} stats={stats} onClose={() => setAboutOpen(false)} />
    </div>
  );
}
