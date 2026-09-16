const API_BASE = (import.meta.env.VITE_API_URL ?? '/api').replace(/\/$/, '');

export async function getResearchRuns(signal) {
  const response = await fetch(`${API_BASE}/research/runs`, { signal });
  if (!response.ok) throw new Error('Research reports could not be loaded.');
  return response.json();
}

export async function getResearchRun(runId, signal) {
  const response = await fetch(`${API_BASE}/research/runs/${encodeURIComponent(runId)}`, { signal });
  if (!response.ok) throw new Error('This research report is unavailable or incomplete.');
  return response.json();
}

export function researchDownloadUrl(runId) {
  return `${API_BASE}/research/runs/${encodeURIComponent(runId)}?download=true`;
}

export async function getDatasetProfile(runId, signal) {
  const response = await fetch(`${API_BASE}/research/runs/${encodeURIComponent(runId)}/profile`, { signal });
  if (response.status === 404) throw new Error('No dataset profile has been saved for this experiment yet.');
  if (!response.ok) throw new Error('The saved dataset profile could not be verified.');
  return response.json();
}

export function profileDownloadUrl(runId) {
  return `${API_BASE}/research/runs/${encodeURIComponent(runId)}/profile?download=true`;
}

export async function getDevelopmentChecks(runId, signal) {
  const response = await fetch(`${API_BASE}/research/runs/${encodeURIComponent(runId)}/cross-validation`, { signal });
  if (response.status === 404) throw new Error('No development checks have been saved for this experiment yet.');
  if (!response.ok) throw new Error('The saved development checks could not be verified.');
  return response.json();
}

export function developmentDownloadUrl(runId) {
  return `${API_BASE}/research/runs/${encodeURIComponent(runId)}/cross-validation?download=true`;
}

export async function getPatients(signal) {
  const res = await fetch(`${API_BASE}/patients`, { signal });
  if (!res.ok) throw new Error('Failed to fetch patients');
  return res.json();
}

export async function getPatientBloodTests(patientId, signal) {
  const res = await fetch(`${API_BASE}/patients/${encodeURIComponent(patientId)}/blood-tests`, { signal });
  if (!res.ok) throw new Error('Failed to fetch blood tests');
  return res.json();
}

export async function getPatientRiskScores(patientId, signal) {
  const res = await fetch(`${API_BASE}/patients/${encodeURIComponent(patientId)}/risk-scores`, { signal });
  if (!res.ok) throw new Error('Failed to fetch risk scores');
  return res.json();
}
