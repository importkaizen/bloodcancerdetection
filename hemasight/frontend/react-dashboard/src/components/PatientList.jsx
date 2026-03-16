import { Link } from 'react-router-dom';
import { getPatients } from '../api/client';
import { useEffect, useState } from 'react';

export function PatientList() {
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    getPatients()
      .then(setPatients)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <p>Loading patients…</p>;
  if (error) return <p>Error: {error}</p>;

  return (
    <div className="patient-list">
      <h2>Patients</h2>
      <ul>
        {patients.map((p) => (
          <li key={p.id}>
            <Link to={`/patients/${p.id}`}>Patient {p.external_id}</Link>
            <span className="muted"> (id: {p.id})</span>
          </li>
        ))}
      </ul>
      {patients.length === 0 && <p>No patients yet. Use POST /blood-test to add data.</p>}
    </div>
  );
}
