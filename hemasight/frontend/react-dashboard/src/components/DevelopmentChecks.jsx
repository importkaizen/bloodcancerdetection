import { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { getDevelopmentChecks, developmentDownloadUrl } from '../api/client';

const MODELS = { prevalence: 'Prevalence baseline', logistic: 'Logistic regression', rf: 'Random Forest', xgboost: 'XGBoost' };
const FEATURES = { basic_cbc: '5 common CBC measurements', expanded_cbc: '15 CBC measurements' };
const num = (n) => n.toLocaleString();
const decimal = (n) => n.toFixed(4);
const mean = (rows, metric) => rows.reduce((sum, r) => sum + r[metric], 0) / rows.length;
const name = (model, control) => control === 'original' ? MODELS[model] : 'Logistic · shuffled training labels';

export function DevelopmentChecks({ runId }) {
  const [report, setReport] = useState(null);
  const [error, setError] = useState('');
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    getDevelopmentChecks(runId, controller.signal)
      .then((data) => { if (!controller.signal.aborted) { setReport(data); setError(''); } })
      .catch((err) => { if (!controller.signal.aborted) setError(err.message); });
    return () => controller.abort();
  }, [runId, attempt]);
  if (error) return <section className="research-panel" role="alert"><h3>Development checks unavailable</h3><p>{error}</p><button onClick={() => setAttempt(attempt + 1)}>Try again</button></section>;
  return report ? <CheckResults report={report} runId={runId}/> : <p role="status">Loading development checks…</p>;
}

function CheckResults({ report, runId }) {
  const [featureSet, setFeatureSet] = useState(Object.keys(report.feature_sets)[0]);
  const [detail, setDetail] = useState('logistic:original');
  const experiments = [...report.models.map((model) => ({model, control:'original'})), {model:'logistic',control:'shuffled_training_labels'}];
  const summary = experiments.map((e) => ({...e, rows: report.results.filter((r) => r.feature_set === featureSet && r.model === e.model && r.control === e.control)}));
  const selected = summary.find((e) => `${e.model}:${e.control}` === detail);
  const rows = [...selected.rows].sort((a,b) => a.fold-b.fold);
  return <>
    <div className="research-notice"><strong>Development-only research diagnostics</strong><p>These folds split records, not verified unique patients. Repeated patients may cross folds. Fold variation is descriptive and can understate uncertainty; it does not establish patient-disjoint performance. No validation, external, or test records were fitted or scored in this check.</p></div>
    <div className="research-controls"><label className="research-field">Measurements for development checks<select value={featureSet} onChange={(e) => setFeatureSet(e.target.value)}>{Object.keys(report.feature_sets).map((key) => <option key={key} value={key}>{FEATURES[key]}</option>)}</select></label><a className="report-download" href={developmentDownloadUrl(runId)}>Download development checks</a></div>
    <div className="research-stats"><div><span>Development records</span><strong>{num(report.development_n)}</strong><small>each evaluated once per experiment</small></div><div><span>Folds</span><strong>{report.n_splits}</strong><small>same stratified splits for every model</small></div><div><span>Positive labels</span><strong>{num(report.development_positive)}</strong><small>{(100*report.development_positive/report.development_n).toFixed(2)}% prevalence</small></div><div><span>Experiment seed</span><strong>{report.seed}</strong><small>fixed before running these checks</small></div></div>
    <section className="research-panel"><h3>Stability across development folds</h3><p className="muted">Each fold fits its own imputation and, for logistic regression, scaling on the other folds. Means give each fold equal weight. The range shows the lowest and highest fold scores, not a confidence interval.</p><div className="table-scroll" tabIndex={0} role="region" aria-label="Cross-validation comparison table"><table className="research-table"><caption>{FEATURES[featureSet]} · {report.n_splits} folds</caption><thead><tr><th scope="col">Experiment</th><th scope="col">Mean AUROC</th><th scope="col">AUROC range</th><th scope="col">Mean average precision</th><th scope="col">Mean Brier score</th></tr></thead><tbody>{summary.map((e) => <tr key={`${e.model}:${e.control}`}><th scope="row">{name(e.model,e.control)}</th><td>{decimal(mean(e.rows,'auroc'))}</td><td>{decimal(Math.min(...e.rows.map((r) => r.auroc)))} – {decimal(Math.max(...e.rows.map((r) => r.auroc)))}</td><td>{decimal(mean(e.rows,'average_precision'))}</td><td>{decimal(mean(e.rows,'brier'))}</td></tr>)}</tbody></table></div><p className="table-note">These are separate development diagnostics, computed after the earlier benchmark. They do not replace external evaluation, select a winner, or justify retuning against already inspected holdouts.</p></section>
    <section className="research-panel"><h3>Shuffled-label sanity check</h3><p>For each fold, logistic regression is fitted after randomly shuffling only that fold’s training labels. Evaluation labels stay unchanged. This breaks the training feature–label pairing while preserving class counts.</p><p className="table-note">Only one shuffle per fold was run. The result is a descriptive control, not a formal permutation test, a p-value, or proof that all leakage is absent. Compare its full fold range rather than interpreting one score as a pass or fail.</p></section>
    <div className="detail-heading"><h3>Inspect each fold</h3><label className="research-field">Development experiment<select value={detail} onChange={(e) => setDetail(e.target.value)}>{experiments.map((e) => <option key={`${e.model}:${e.control}`} value={`${e.model}:${e.control}`}>{name(e.model,e.control)}</option>)}</select></label></div>
    <div className="research-chart-grid"><section className="research-panel"><h3>AUROC by fold</h3><p className="muted">{name(selected.model,selected.control)} · {FEATURES[featureSet]}</p><ResponsiveContainer width="100%" height={270}><LineChart data={rows} margin={{top:10,right:15,left:0,bottom:10}}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="fold" tickFormatter={(v) => `Fold ${v}`}/><YAxis domain={[0,1]}/><Tooltip formatter={(v) => decimal(Number(v))} labelFormatter={(v) => `Fold ${v}`}/><ReferenceLine y={.5} stroke="#899ead" strokeDasharray="5 5"/><Line type="linear" dataKey="auroc" name="AUROC" stroke="#24658a" strokeWidth={2} dot={{r:4}} isAnimationActive={false}/></LineChart></ResponsiveContainer><p className="table-note">Dashed line: AUROC 0.5. Lines connect folds for readability; folds are not a time series.</p></section><section className="research-panel"><h3>Fold results</h3><div className="table-scroll" tabIndex={0} role="region" aria-label="Individual fold results"><table className="research-table"><thead><tr><th scope="col">Fold</th><th scope="col">Training records</th><th scope="col">Evaluation records</th><th scope="col">AUROC</th><th scope="col">Average precision</th><th scope="col">Brier</th></tr></thead><tbody>{rows.map((r) => <tr key={r.fold}><th scope="row">{r.fold}</th><td>{num(r.train_n)}</td><td>{num(r.evaluation_n)}</td><td>{decimal(r.auroc)}</td><td>{decimal(r.average_precision)}</td><td>{decimal(r.brier)}</td></tr>)}</tbody></table></div></section></div>
    <details className="research-panel provenance"><summary>Protocol and reproducibility</summary><p>Shuffled stratified K-fold splits use seed {report.seed}. Estimator seeds and training-label shuffle seeds are the base seed plus the fold number. Class counts are preserved approximately by the splitter. Model settings use the existing fixed baseline pipelines. No operating threshold is selected, and no models or record-level predictions are saved.</p><p>Patient IDs and dates are unavailable. Stratification can make folds more homogeneous, and overlapping training sets prevent treating fold scores as independent replicates. This view reports descriptive fold ranges only.</p><dl><dt>Completed</dt><dd>{report.completed_at_utc.slice(0,10)}</dd><dt>Dataset SHA-256</dt><dd className="hash-value">{report.dataset_sha256}</dd><dt>Software versions</dt><dd>{Object.entries(report.versions).map(([k,v]) => `${k} ${v}`).join(' · ')}</dd></dl><p>Source hashes and every fold’s aggregate metrics are included in the download. See the <a href="https://scikit-learn.org/stable/modules/cross_validation.html" target="_blank" rel="noreferrer">scikit-learn cross-validation guide</a> for the method and its limitations.</p></details>
  </>;
}
