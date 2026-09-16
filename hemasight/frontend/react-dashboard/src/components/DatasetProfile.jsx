import { useEffect, useState } from 'react';
import { getDatasetProfile, profileDownloadUrl } from '../api/client';

const NAMES = { wbc: 'White blood cells', rbc: 'Red blood cells', platelets: 'Platelets', hemoglobin: 'Hemoglobin', lymphocytes: 'Lymphocytes (%)', hematocrit: 'Hematocrit', mcv: 'Mean cell volume', mch: 'Mean cell hemoglobin', mchc: 'Mean cell hemoglobin concentration', rdw_cv: 'Red cell distribution width', neutrophils: 'Neutrophils', monocytes: 'Monocytes', eosinophils: 'Eosinophils', basophils: 'Basophils', lymphocytes_absolute: 'Lymphocytes (absolute)' };
const COHORTS = { development: 'Development', validation: 'Validation', external: 'External', test: 'Test file' };
const numeric = (value) => value == null ? '—' : value.toLocaleString(undefined, { maximumFractionDigits: 3 });
const pct = (value) => `${value.toFixed(2)}%`;
const groupName = (group) => group.scope === 'overall' ? COHORTS[group.cohort] : group.site === 'real_world' ? 'Test file' : `Hospital ${group.site}`;

function DistributionPlot({ groups, feature, unit }) {
  const available = groups.map((g) => g.features[feature]).filter((d) => d.observed > 0);
  if (!available.length) return <p>No observed values are available for this measurement.</p>;
  const low = Math.min(...available.map((d) => d.p05));
  const high = Math.max(...available.map((d) => d.p95));
  const x = (value) => high === low ? 410 : 140 + 540 * (value - low) / (high - low);
  const height = groups.length * 52 + 65;
  return <div className="table-scroll" tabIndex={0} role="region" aria-label="Measurement spread chart"><svg viewBox={`0 0 730 ${height}`} role="img" aria-label={`${NAMES[feature]} distribution in ${unit}: lines show 5th–95th percentiles, boxes the middle 50%, dots the median.`} style={{width:'100%',minWidth:400}}>
    {groups.map((g, index) => { const d = g.features[feature]; const y = index * 52 + 30; return <g key={`${g.cohort}-${g.site}`}><text x={125} y={y + 4} textAnchor="end" fontSize={15} fill="#465d6b">{groupName(g)}</text>{d.observed ? <g><line x1={x(d.p05)} x2={x(d.p95)} y1={y} y2={y} stroke="#7696a8" strokeWidth={2}/><line x1={x(d.p05)} x2={x(d.p05)} y1={y-7} y2={y+7} stroke="#7696a8"/><line x1={x(d.p95)} x2={x(d.p95)} y1={y-7} y2={y+7} stroke="#7696a8"/><rect x={x(d.q1)} y={y-10} width={Math.max(1,x(d.q3)-x(d.q1))} height={20} rx={3} fill="#d0e5ec"/><circle cx={x(d.median)} cy={y} r={5} fill="#24658a"/></g> : <text x={140} y={y+4} fontSize={14} fill="#526575">No observed values</text>}</g>; })}
    <line x1={140} x2={680} y1={height-38} y2={height-38} stroke="#b7c8d3"/>
    {[0,.25,.5,.75,1].map((fraction) => <text key={fraction} x={140+540*fraction} y={height-15} textAnchor="middle" fontSize={13} fill="#526575">{high === low ? (fraction === .5 ? numeric(low) : '') : numeric(low+(high-low)*fraction)}</text>)}
  </svg></div>;
}

export function DatasetProfile({ runId }) {
  const [profile, setProfile] = useState(null);
  const [error, setError] = useState('');
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    getDatasetProfile(runId, controller.signal)
      .then((data) => { if (!controller.signal.aborted) { setProfile(data); setError(''); } })
      .catch((err) => { if (!controller.signal.aborted) setError(err.message); });
    return () => controller.abort();
  }, [runId, attempt]);
  if (error) return <section className="research-panel" role="alert"><h3>Dataset comparison unavailable</h3><p>{error}</p><button onClick={() => setAttempt(attempt + 1)}>Try again</button></section>;
  return profile ? <ProfileResults profile={profile} runId={runId} /> : <p role="status">Loading dataset comparison…</p>;
}

function ProfileResults({ profile, runId }) {
  const [feature, setFeature] = useState('wbc');
  const [scope, setScope] = useState('cohorts');
  const pooled = profile.groups.filter((g) => g.scope === 'overall');
  const groups = profile.groups.filter((g) => scope === 'cohorts' ? g.scope === 'overall' : g.scope === 'site' && g.cohort === scope);
  const reference = pooled.find((g) => g.cohort === 'development').features[feature];
  const missing = groups.reduce((sum, g) => sum + g.features[feature].missing, 0);
  return <>
    <div className="research-notice"><strong>Understand the dataset behind the scores</strong><p>Compare recorded measurements before imputation or scaling. Differences may reflect case mix, collection practices, or other factors; this view cannot identify their cause or establish clinical validity.</p></div>
    <div className="research-controls">
      <label className="research-field">Measurement<select value={feature} onChange={(e) => setFeature(e.target.value)}>{Object.keys(profile.units).map((name) => <option value={name} key={name}>{NAMES[name]}</option>)}</select></label>
      <label className="research-field">Compare groups<select value={scope} onChange={(e) => setScope(e.target.value)}><option value="cohorts">All evaluated cohorts + development</option>{pooled.map((g) => <option value={g.cohort} key={g.cohort}>{g.cohort === 'test' ? 'Released test file' : `${COHORTS[g.cohort]} hospitals`}</option>)}</select></label>
      <a className="report-download" href={profileDownloadUrl(runId)}>Download dataset profile</a>
    </div>
    <section className="research-panel">
      <div className="panel-heading"><h3>{NAMES[feature]} <span className="muted">({profile.units[feature]})</span></h3><span className="muted">Development median: {numeric(reference.median)}</span></div>
      <p className="muted">The middle 50% spans the 25th to 75th percentile. The 5th–95th interval describes the central 90% of observed values; it is not a clinical reference range or confidence interval.</p>
      <div className="table-scroll" tabIndex={0} role="region" aria-label="Measurement distribution table"><table className="research-table"><caption>Observed {NAMES[feature].toLowerCase()} values in {profile.units[feature]}</caption><thead><tr><th scope="col">Group</th><th scope="col">Records</th><th scope="col">Observed</th><th scope="col">Missing</th><th scope="col">Median</th><th scope="col">Middle 50%</th><th scope="col">5th–95th percentile</th></tr></thead><tbody>{groups.map((g) => { const d = g.features[feature]; return <tr key={`${g.cohort}-${g.site}`}><th scope="row">{groupName(g)}</th><td>{numeric(g.n)}</td><td>{numeric(d.observed)}</td><td>{numeric(d.missing)} ({pct(100*d.missing/g.n)})</td><td>{numeric(d.median)}</td><td>{d.observed ? `${numeric(d.q1)} – ${numeric(d.q3)}` : '—'}</td><td>{d.observed ? `${numeric(d.p05)} – ${numeric(d.p95)}` : '—'}</td></tr>; })}</tbody></table></div>
      <p className="table-note">Quantiles use observed values only, with linear interpolation. Missing measurements stay missing. Undefined summaries are shown as —. Sample records may include repeated patients.</p>
    </section>
    <div className="research-chart-grid">
      <section className="research-panel"><h3>Measurement spread</h3><p className="muted">Lines: 5th–95th percentiles. Boxes: middle 50%. Dots: median. All groups share the same measurement scale ({profile.units[feature]}).</p><DistributionPlot groups={groups} feature={feature} unit={profile.units[feature]}/><p className="table-note">{missing === 0 ? 'No missing values for this measurement in the selected prepared records.' : `${numeric(missing)} missing measurements excluded from these summaries; see group rates above.`}</p></section>
      <section className="research-panel"><h3>Case mix in these groups</h3><p className="muted">The proportion of positive labels differs across cohorts and can affect precision and probability calibration.</p><div className="table-scroll" tabIndex={0} role="region" aria-label="Dataset case mix table"><table className="research-table"><thead><tr><th scope="col">Group</th><th scope="col">Positive labels</th><th scope="col">Prevalence</th></tr></thead><tbody>{groups.map((g) => <tr key={`${g.cohort}-${g.site}`}><th scope="row">{groupName(g)}</th><td>{numeric(g.n_positive)}</td><td>{pct(100*g.n_positive/g.n)}</td></tr>)}</tbody></table></div><p className="table-note">Labels describe the released record. They do not define future cancer incidence.</p></section>
    </div>
    <details className="research-panel provenance"><summary>How this comparison was made</summary><p>These aggregate summaries come from the exact prepared CSV used by the saved benchmark. No model was trained, no predictions were generated, and no missing values were filled. Only development and previously evaluated cohorts are included.</p><p>Possible patient overlap and measurement dates cannot be checked. Observed differences are descriptive, with no significance tests or patient-level uncertainty estimates.</p><dl><dt>Generated</dt><dd>{profile.generated_at_utc.slice(0,10)}</dd><dt>Dataset SHA-256</dt><dd className="hash-value">{profile.dataset_sha256}</dd><dt>Profile code SHA-256</dt><dd className="hash-value">{profile.source_sha256}</dd></dl></details>
  </>;
}
