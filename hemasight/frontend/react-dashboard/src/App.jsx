import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import { lazy, Suspense } from 'react';
import { PatientList } from './components/PatientList';
import './App.css';

const PatientDetail = lazy(() => import('./components/PatientDetail').then((module) => ({ default: module.PatientDetail })));
const ResearchDashboard = lazy(() => import('./components/ResearchDashboard').then((module) => ({ default: module.ResearchDashboard })));

function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <header>
          <h1>HemaSight</h1>
          <p className="tagline">Blood-pattern research workspace</p>
          <p className="muted">Experimental model outputs for research; not diagnostic results.</p>
          <nav aria-label="Workspace"><NavLink to="/research">Research benchmarks</NavLink><NavLink to="/" end>Patient workspace</NavLink></nav>
        </header>
        <main>
          <Suspense fallback={<p role="status">Loading workspace…</p>}>
          <Routes>
            <Route path="/" element={<PatientList />} />
            <Route path="/patients/:patientId" element={<PatientDetail />} />
            <Route path="/research" element={<ResearchDashboard />} />
          </Routes>
          </Suspense>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
