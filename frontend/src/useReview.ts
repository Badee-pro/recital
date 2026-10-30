import { useCallback, useEffect, useRef, useState } from "react";
import { api, type Flag, type FlagStatus, type Job, type Jobs } from "./api";

const isActive = (j: Job) => j.state === "queued" || j.state === "running";

/** Flags in the review queue and the background review jobs, polled while any job is running. */
export function useReview(refreshContracts: () => Promise<void>) {
  const [flags, setFlags] = useState<Flag[]>([]);
  const [jobs, setJobs] = useState<Jobs>({});
  const lastJobs = useRef("");
  const timer = useRef<number>(undefined);

  const loadFlags = useCallback(async () => {
    setFlags((await api<{ flags: Flag[] }>("/api/flags")).flags);
  }, []);

  const pollJobs = useCallback(async () => {
    window.clearTimeout(timer.current);
    let next: Jobs;
    try { next = await api<Jobs>("/api/review/status"); } catch { return; }
    setJobs(next);
    const key = JSON.stringify(next);
    if (key !== lastJobs.current) {  // a job moved on: new flags or counts may exist
      lastJobs.current = key;
      loadFlags();
      refreshContracts();
    }
    if (Object.values(next).some(isActive)) timer.current = window.setTimeout(pollJobs, 1500);
  }, [loadFlags, refreshContracts]);

  useEffect(() => {
    loadFlags();
    pollJobs();
    return () => window.clearTimeout(timer.current);
  }, [loadFlags, pollJobs]);

  const runReview = useCallback(async (docIds: string | string[]) => {
    await Promise.all([docIds].flat().map((id) => api<Job>(`/api/review/${id}`, {})));
    pollJobs();
  }, [pollJobs]);

  const decide = useCallback(async (id: string, status: FlagStatus, reviewer: string, note: string) => {
    await api<Flag>(`/api/flags/${id}`, { status, reviewer, note });
    await Promise.all([loadFlags(), refreshContracts()]);
  }, [loadFlags, refreshContracts]);

  return {
    flags, jobs, loadFlags, pollJobs, runReview, decide,
    openCount: flags.filter((f) => f.status === "open").length,
  };
}

export type Review = ReturnType<typeof useReview>;
export { isActive };
