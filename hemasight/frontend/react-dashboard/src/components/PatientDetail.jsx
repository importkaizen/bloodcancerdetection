import { useParams, Link } from 'react-router-dom';
import { useEffect, useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { getPatientBloodTests, getPatientRiskScores } from '../api/client';

function formatDate(d) {
  return new Date(d).toLocaleDateString();
}

export function PatientDetail() {
  const { patientId } = useParams();
  const [bloodTests, setBloodTests] = useState([]);
  const [riskScores, setRiskScores] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!patientId) return;
    Promise.all([
      getPatientBloodTests(patientId),
      getPatientRiskScores(patientId),
    ])
      .then(([bt, rs]) => {
        setBloodTests(bt);
        setRiskScores(rs);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [patientId]);

  if (loading) return <p>Loading…</p>;
  if (error) return <p>Error: {error}</p>;

  const bloodChartData = bloodTests.map((t) => ({
    date: formatDate(t.date),
    full: t.date,
    WBC: t.wbc,
    RBC: t.rbc,
    Platelets: t.platelets,
    Hemoglobin: t.hemoglobin,
    Lymphocytes: t.lymphocytes,
  }));

  const riskChartData = riskScores.map((r) => ({
    date: formatDate(r.computed_at),
    score: r.score,
    level: r.level,
  }));

  return (
    <div className="patient-detail">
      <p><Link to="/">← Back to patients</Link></p>
      <h2>Patient {patientId}</h2>

      <section className="chart-section">
        <h3>Blood metrics over time</h3>
        {bloodChartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={bloodChartData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="WBC" stroke="#8884d8" name="WBC" />
              <Line type="monotone" dataKey="Platelets" stroke="#82ca9d" name="Platelets" />
              <Line type="monotone" dataKey="Hemoglobin" stroke="#ffc658" name="Hemoglobin" />
              <Line type="monotone" dataKey="Lymphocytes" stroke="#ff7c7c" name="Lymphocytes" />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p>No blood tests yet.</p>
        )}
      </section>

      <section className="chart-section">
        <h3>Risk score over time</h3>
        {riskChartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={riskChartData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="score" stroke="#8884d8" name="Risk score" />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p>No risk scores yet. Train the model and process blood tests to see risk trend.</p>
        )}
      </section>

      {riskScores.length > 0 && (
        <section>
          <h3>Latest risk</h3>
          <p><strong>Level:</strong> {riskScores[riskScores.length - 1].level}</p>
          <p><strong>Message:</strong> {riskScores[riskScores.length - 1].message}</p>
        </section>
      )}
    </div>
  );
}
