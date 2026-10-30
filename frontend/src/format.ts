// Display helpers shared by the components.

export const pct = (v: number | undefined) => (v == null ? "–" : Math.round(v * 100) + "%");

const SHORT: Record<string, string> = { nda: "NDA", msa: "MSA", dpa: "DPA", saas: "SaaS" };
const KIND: Record<string, string> = { nda: "Mutual NDA", msa: "Master services agreement", dpa: "Data processing addendum", saas: "SaaS subscription" };
export const METHOD: Record<string, string> = { dense: "Embeddings", hybrid: "Hybrid", bm25: "Keyword" };

export const kindOf = (docId: string) => docId.split("_")[0];
export const shortOf = (docId: string) => SHORT[kindOf(docId)];
export const kindLabel = (docId: string) => KIND[kindOf(docId)] || "Agreement";
// "MSA — Northwind Logistics Corp. and Quillon Software, Inc." -> "Northwind Logistics and Quillon Software"
export const partiesOf = (title: string) => (title.split(" — ")[1] || title).replace(/, Inc\.|, LLC| LLC| Corp\.| Inc\./g, "");
// -> "Northwind & Quillon"
export const pairOf = (title: string) => (title.split(" — ")[1] || title).split(" and ").map((p) => p.trim().split(" ")[0]).join(" & ");
export const refOf = (c: { doc_id: string; clause: string }) => `${shortOf(c.doc_id) || "Doc"} §${c.clause}`;
export const methodLabel = (m: string) => METHOD[m] || m;
