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

const METRICS = {
  WBC: 'WBC (K/µL)', RBC: 'RBC (M/µL)', Platelets: 'Platelets (K/µL)',
  Hemoglobin: 'Hemoglobin (g/dL)', Lymphocytes: 'Lymphocytes (%)',
};

function formatDate(d) {
  if (typeof d === 'number') return new Date(d).toISOString().slice(0, 10);
  return d ? d.slice(0, 10) : 'Unknown date';
}

export function PatientDetail() {
  const { patientId } = useParams();
  return <PatientView key={patientId} patientId={patientId} />;
}

function PatientView({ patientId }) {
  const [bloodTests, setBloodTests] = useState([]);
  const [riskScores, setRiskScores] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [metric, setMetric] = useState('WBC');

  useEffect(() => {
    if (!patientId) return;
    const controller = new AbortController();
    Promise.all([
      getPatientBloodTests(patientId, controller.signal),
      getPatientRiskScores(patientId, controller.signal),
    ])
      .then(([bt, rs]) => {
        if (controller.signal.aborted) return;
        setBloodTests(bt);
        setRiskScores(rs);
      })
      .catch((e) => { if (!controller.signal.aborted) setError(e.message); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [patientId]);

  if (loading) return <p>Loading…</p>;
  if (error) return <p>Error: {error}</p>;

  const bloodChartData = bloodTests.map((t) => ({
    date: Date.parse(`${t.date.slice(0, 10)}T00:00:00Z`),
    full: t.date,
    WBC: t.wbc,
    RBC: t.rbc,
    Platelets: t.platelets,
    Hemoglobin: t.hemoglobin,
    Lymphocytes: t.lymphocytes,
  }));

  const datedScores = riskScores.filter((r) => r.blood_test_date);
  const riskChartData = datedScores.map((r) => ({
    date: Date.parse(`${r.blood_test_date.slice(0, 10)}T00:00:00Z`),
    score: r.score,
    level: r.level,
  }));

  return (
    <div className="patient-detail">
      <p><Link to="/">← Back to patients</Link></p>
      <h2>Patient {patientId}</h2>

      <section className="chart-section">
        <h3>Blood metrics over time</h3>
        <div className="metric-picker">
          <label htmlFor="blood-metric">Measurement</label>
          <select id="blood-metric" value={metric} onChange={(e) => setMetric(e.target.value)}>
            {Object.entries(METRICS).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
          </select>
        </div>
        {bloodChartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={bloodChartData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" type="number" scale="time" domain={['dataMin', 'dataMax']} tickFormatter={formatDate} minTickGap={30} />
              <YAxis />
              <Tooltip labelFormatter={formatDate} />
              <Legend />
              <Line type="linear" dataKey={metric} stroke="#155b88" strokeWidth={2} name={METRICS[metric]} connectNulls={false} />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p>No blood tests yet.</p>
        )}
      </section>

      <section className="chart-section">
        <h3>Experimental score by blood-test date</h3>
        {riskChartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={riskChartData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" type="number" scale="time" domain={['dataMin', 'dataMax']} tickFormatter={formatDate} minTickGap={30} />
              <YAxis domain={[0, 1]} />
              <Tooltip labelFormatter={formatDate} />
              <Legend />
              <Line type="linear" dataKey="score" stroke="#155b88" strokeWidth={2} name="Experimental score" />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p>No dated model outputs are available for this patient.</p>
        )}
        {datedScores.length < riskScores.length && <p>Some older outputs have no linked blood-test date and are omitted from the timeline.</p>}
      </section>

      {datedScores.length > 0 && (
        <section>
          <h3>Latest dated model output</h3>
          <p><strong>Blood-test date:</strong> {formatDate(datedScores[datedScores.length - 1].blood_test_date)}</p>
          <p><strong>Experimental level:</strong> {datedScores[datedScores.length - 1].level}</p>
          <p><strong>Model:</strong> {datedScores[datedScores.length - 1].model_version}</p>
        </section>
      )}
    </div>
  );
}
