import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { PatientList } from './components/PatientList';
import { PatientDetail } from './components/PatientDetail';
import './App.css';

function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <header>
          <h1>HemaSight</h1>
          <p className="tagline">Early hematologic abnormality risk from CBC time-series</p>
        </header>
        <main>
          <Routes>
            <Route path="/" element={<PatientList />} />
            <Route path="/patients/:patientId" element={<PatientDetail />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
