const API_BASE = import.meta.env.VITE_API_URL ?? '';

export async function getPatients() {
  const res = await fetch(`${API_BASE}/patients`);
  if (!res.ok) throw new Error('Failed to fetch patients');
  return res.json();
}

export async function getPatientBloodTests(patientId) {
  const res = await fetch(`${API_BASE}/patients/${patientId}/blood-tests`);
  if (!res.ok) throw new Error('Failed to fetch blood tests');
  return res.json();
}

export async function getPatientRiskScores(patientId) {
  const res = await fetch(`${API_BASE}/patients/${patientId}/risk-scores`);
  if (!res.ok) throw new Error('Failed to fetch risk scores');
  return res.json();
}
