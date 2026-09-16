import { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter, ReferenceLine,
} from 'recharts';
import { getResearchRuns, getResearchRun, researchDownloadUrl } from '../api/client';
import './ResearchDashboard.css';
import { DatasetProfile } from './DatasetProfile';
import { DevelopmentChecks } from './DevelopmentChecks';

const MODEL_NAMES = { prevalence: 'Prevalence baseline', logistic: 'Logistic regression', rf: 'Random Forest', xgboost: 'XGBoost' };
const FEATURE_NAMES = { basic_cbc: '5 common CBC measurements', expanded_cbc: '15 CBC measurements' };
const COHORT_NAMES = { validation: 'Validation · hospitals A–C', external: 'External · hospitals D–G', test: 'Released real-world test file' };
const number = (value) => value == null ? '—' : value.toLocaleString();
const decimal = (value) => value == null ? '—' : value.toFixed(4);
const percent = (value) => value == null ? '—' : `${(value * 100).toFixed(2)}%`;

function Loading() { return <p role="status">Loading research results…</p>; }

export function ResearchDashboard() {
  const [catalog, setCatalog] = useState(null);
  const [chosenRun, setChosenRun] = useState('');
  const [error, setError] = useState('');
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    getResearchRuns(controller.signal)
      .then((data) => { if (!controller.signal.aborted) { setCatalog(data); setError(''); } })
      .catch((err) => { if (!controller.signal.aborted) setError(err.message); });
    return () => controller.abort();
  }, [attempt]);
  if (error) return <div role="alert"><p>{error}</p><button onClick={() => setAttempt(attempt + 1)}>Try again</button></div>;
  if (!catalog) return <Loading />;
  if (!catalog.runs.length) return <section className="research-panel"><h2>Research benchmarks</h2><p>No completed reports are available.</p>{catalog.unavailable_runs > 0 && <p>Some report files are incomplete or invalid.</p>}</section>;
  const runId = catalog.runs.some((run) => run.run_id === chosenRun) ? chosenRun : catalog.runs[0].run_id;
  return <div className="research-dashboard">
    <div className="research-heading">
      <div><p className="eyebrow">CBC SCREENING STUDY</p><h2>Research benchmarks</h2><p className="muted">Explore the completed experiments across the released hospital cohorts.</p></div>
      <label className="research-field">Saved experiment
        <select value={runId} onChange={(event) => setChosenRun(event.target.value)}>
          {catalog.runs.map((run) => <option key={run.run_id} value={run.run_id}>{run.run_id} · {run.completed_at_utc.slice(0, 10)}</option>)}
        </select>
      </label>
    </div>
    {catalog.unavailable_runs > 0 && <p role="status">{catalog.unavailable_runs} other report(s) could not be loaded.</p>}
    <RunLoader key={runId} runId={runId} />
  </div>;
}

function RunLoader({ runId }) {
  const [searchParams, setSearchParams] = useSearchParams();
  const view = ['dataset', 'development'].includes(searchParams.get('view')) ? searchParams.get('view') : 'performance';
  const setView = (next) => setSearchParams((previous) => {
    const params = new URLSearchParams(previous);
    if (next !== 'performance') params.set('view', next);
    else params.delete('view');
    return params;
  });
  const [report, setReport] = useState(null);
  const [error, setError] = useState('');
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    getResearchRun(runId, controller.signal)
      .then((data) => { if (!controller.signal.aborted) { setReport(data); setError(''); } })
      .catch((err) => { if (!controller.signal.aborted) setError(err.message); });
    return () => controller.abort();
  }, [runId, attempt]);
  if (error) return <div role="alert"><p>{error}</p><button onClick={() => setAttempt(attempt + 1)}>Try again</button></div>;
  return report ? <><div className="research-view-switch" role="group" aria-label="Research view"><button aria-pressed={view === 'performance'} onClick={() => setView('performance')}>Model performance</button><button aria-pressed={view === 'dataset'} onClick={() => setView('dataset')}>Dataset comparison</button><button aria-pressed={view === 'development'} onClick={() => setView('development')}>Development checks</button></div>{view === 'performance' ? <Results report={report} /> : view === 'dataset' ? <DatasetProfile runId={runId} /> : <DevelopmentChecks runId={runId} />}</> : <Loading />;
}

function CalibrationTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const bin = payload[0].payload;
  return <div className="chart-tooltip"><strong>{number(bin.n)} records in bin</strong><br />Mean prediction: {percent(bin.predicted)}<br />Observed positive fraction: {percent(bin.observed)}</div>;
}

function Results({ report }) {
  const { manifest, results, run_id: runId } = report;
  const [cohort, setCohort] = useState(manifest.evaluated_cohorts.includes('external') ? 'external' : manifest.evaluated_cohorts[0]);
  const [featureSet, setFeatureSet] = useState(Object.keys(manifest.feature_sets)[0]);
  const [model, setModel] = useState(manifest.models.find((name) => name !== 'prevalence') ?? manifest.models[0]);
  const overall = results.filter((row) => row.cohort === cohort && row.feature_set === featureSet && row.scope === 'overall');
  const selected = overall.find((row) => row.model === model);
  const sites = results.filter((row) => row.cohort === cohort && row.feature_set === featureSet && row.model === model && row.scope === 'site');
  const counts = manifest.cohort_counts[cohort];
  const siteChart = sites.map((row) => ({ site: row.site === 'real_world' ? 'Test file' : row.site, sensitivity: row.metrics.sensitivity == null ? null : row.metrics.sensitivity * 100, specificity: row.metrics.specificity == null ? null : row.metrics.specificity * 100 }));
  const bins = selected.reliability_bins.filter((bin) => bin.n > 0 && bin.mean_predicted_probability != null && bin.observed_positive_fraction != null).map((bin) => ({ n: bin.n, predicted: bin.mean_predicted_probability, observed: bin.observed_positive_fraction }));
  const provenance = manifest.preparation_provenance;
  return <>
    <div className="research-notice">
      <strong>Record-level screening results</strong>
      <p>Patient identifiers and diagnosis dates are unavailable. These results cannot establish patient-disjoint performance or prediction before diagnosis. {manifest.evaluate_holdouts ? 'External and test results have already been inspected.' : 'This run contains validation results only.'}</p>
    </div>
    <div className="research-controls">
      <label className="research-field">Evaluation cohort<select value={cohort} onChange={(event) => setCohort(event.target.value)}>{manifest.evaluated_cohorts.map((name) => <option key={name} value={name}>{COHORT_NAMES[name]}</option>)}</select></label>
      <label className="research-field">Measurement set<select value={featureSet} onChange={(event) => setFeatureSet(event.target.value)}>{Object.keys(manifest.feature_sets).map((name) => <option key={name} value={name}>{FEATURE_NAMES[name]}</option>)}</select></label>
      <a className="report-download" href={researchDownloadUrl(runId)}>Download aggregate results</a>
    </div>
    <div className="research-stats" aria-label="Selected cohort counts">
      <div><span>Records evaluated</span><strong>{number(counts.n)}</strong><small>in this cohort</small></div>
      <div><span>Positive labels</span><strong>{number(counts.n_positive)}</strong><small>{percent(counts.n_positive / counts.n)} prevalence</small></div>
      <div><span>Development records</span><strong>{number(manifest.cohort_counts.development.n)}</strong><small>used to fit each model</small></div>
      <div><span>Specificity target</span><strong>{percent(manifest.target_specificity)}</strong><small>selected on pooled validation</small></div>
    </div>
    <section className="research-panel">
      <div className="panel-heading"><h3>Model comparison</h3><span className="muted">Fixed settings · seed {manifest.seed}</span></div>
      <p className="muted">All planned models are shown. The validation-selected threshold stays fixed across hospitals and test records.</p>
      {cohort === 'validation' && <p className="inline-note">Validation records also selected the threshold, so these operating-point results are not an unbiased performance estimate.</p>}
      <div className="table-scroll" tabIndex={0} role="region" aria-label="Model comparison table">
        <table className="research-table"><caption>{COHORT_NAMES[cohort]} · {FEATURE_NAMES[featureSet]}</caption>
          <thead><tr><th scope="col">Model</th><th scope="col">AUROC</th><th scope="col">Average precision</th><th scope="col">Sensitivity</th><th scope="col">Specificity</th><th scope="col">Precision</th><th scope="col">Brier score</th><th scope="col">FP / 1,000 negatives</th></tr></thead>
          <tbody>{overall.map((row) => <tr key={row.model}><th scope="row">{MODEL_NAMES[row.model]}</th><td>{decimal(row.metrics.auroc)}</td><td>{decimal(row.metrics.average_precision)}</td><td>{percent(row.metrics.sensitivity)}</td><td>{percent(row.metrics.specificity)}</td><td>{percent(row.metrics.precision)}</td><td>{decimal(row.metrics.brier)}</td><td>{row.metrics.false_positives_per_1000_negatives == null ? '—' : row.metrics.false_positives_per_1000_negatives.toFixed(1)}</td></tr>)}</tbody>
        </table>
      </div>
      <p className="table-note">AUROC measures ranking; average precision depends on prevalence. Lower Brier scores indicate better probability estimates. FP = false positives. Undefined metrics are shown as —.</p>
    </section>
    <div className="detail-heading"><h3>Look closer</h3><label className="research-field">Model for detailed charts<select value={model} onChange={(event) => setModel(event.target.value)}>{manifest.models.map((name) => <option key={name} value={name}>{MODEL_NAMES[name]}</option>)}</select></label></div>
    <div className="research-chart-grid">
      <section className="research-panel">
        <h3>Performance by hospital</h3>
        <p className="muted">A pooled threshold can behave differently at each site.</p>
        <ResponsiveContainer width="100%" height={280}><BarChart data={siteChart} margin={{top:15,right:10,left:0,bottom:5}}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="site" /><YAxis domain={[0,100]} tickFormatter={(value) => `${value}%`} /><Tooltip formatter={(value) => value == null ? 'Undefined' : `${Number(value).toFixed(2)}%`} /><Legend /><Bar dataKey="sensitivity" name="Sensitivity" fill="#24658a" radius={[3,3,0,0]} /><Bar dataKey="specificity" name="Specificity" fill="#258574" radius={[3,3,0,0]} /></BarChart></ResponsiveContainer>
      </section>
      <section className="research-panel">
        <h3>Probability calibration</h3>
        <p className="muted">Each dot is a nonempty probability bin. The dashed line is perfect agreement.</p>
        <ResponsiveContainer width="100%" height={280}><ScatterChart margin={{top:15,right:25,left:5,bottom:25}}><CartesianGrid strokeDasharray="3 3" /><XAxis type="number" dataKey="predicted" domain={[0,1]} name="Mean prediction" tickFormatter={(value) => `${Math.round(value*100)}%`} label={{value:'Mean predicted probability',position:'bottom',offset:5}} /><YAxis type="number" dataKey="observed" domain={[0,1]} name="Observed fraction" tickFormatter={(value) => `${Math.round(value*100)}%`} /><Tooltip content={<CalibrationTooltip />} /><ReferenceLine segment={[{x:0,y:0},{x:1,y:1}]} stroke="#899ead" strokeDasharray="5 5" /><Scatter data={bins} fill="#24658a" /></ScatterChart></ResponsiveContainer>
        <p className="table-note">Vertical axis: observed positive fraction. Bin sizes vary; the chart does not show uncertainty intervals.</p>
      </section>
    </div>
    <section className="research-panel">
      <div className="panel-heading"><h3>{MODEL_NAMES[model]} · site details</h3><span className="muted">Threshold: {selected.metrics.threshold.toPrecision(6)}</span></div>
      <div className="table-scroll" tabIndex={0} role="region" aria-label="Hospital results table"><table className="research-table"><caption>Counts and operating-point results for {COHORT_NAMES[cohort]}</caption><thead><tr><th scope="col">Site</th><th scope="col">Records</th><th scope="col">Positive labels</th><th scope="col">Prevalence</th><th scope="col">Sensitivity</th><th scope="col">Specificity</th><th scope="col">False positives</th><th scope="col">False negatives</th></tr></thead><tbody>{sites.map((row) => <tr key={row.site}><th scope="row">{row.site === 'real_world' ? 'Test file' : `Hospital ${row.site}`}</th><td>{number(row.metrics.n)}</td><td>{number(row.metrics.n_positive)}</td><td>{percent(row.metrics.prevalence)}</td><td>{percent(row.metrics.sensitivity)}</td><td>{percent(row.metrics.specificity)}</td><td>{number(row.metrics.false_positives)}</td><td>{number(row.metrics.false_negatives)}</td></tr>)}</tbody></table></div>
    </section>
    <details className="research-panel provenance"><summary>Study methods and reproducibility</summary>
      <p>Models and median imputation were fitted on development records only. Logistic regression also used development-fitted scaling. Thresholds targeted {percent(manifest.target_specificity)} specificity on validation negatives. No probability recalibration or hyperparameter search was performed.</p>
      <p>Patient overlap cannot be checked. These are descriptive record-level metrics; patient-cluster confidence intervals and clinical-utility claims are unavailable. New model selection needs a new untouched evaluation cohort.</p>
      <dl><dt>Total released records used</dt><dd>{number(manifest.dataset.rows)}</dd><dt>Completed</dt><dd>{manifest.completed_at_utc.slice(0,10)}</dd><dt>Included measurements</dt><dd>{manifest.feature_sets[featureSet].join(', ')}</dd><dt>Software versions</dt><dd>{Object.entries(manifest.versions).map(([name,version]) => `${name} ${version}`).join(' · ')}</dd><dt>Dataset SHA-256</dt><dd className="hash-value">{manifest.dataset.sha256}</dd></dl>
      {provenance && <p>Source: LeukoAlert v1, DOI {provenance.source_doi}. License: {provenance.source_license}. The release has {number(provenance.difference_from_published_total)} more records than the paper’s stated total; that discrepancy is preserved in the audit.</p>}
      <p className="muted">The downloadable report contains aggregate metrics, reliability bins, and source hashes. Individual records and fitted models are not included.</p>
    </details>
  </>;
}
