import "./App.css";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import LandingPage from "./components/LandingPage";
import SAEVisualizer from "./SAEVisualizer";
import ErrorBoundary from "./components/ErrorBoundary";

function App() {
  return (
    <ErrorBoundary>
      <Router>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/sae-visualizer" element={<SAEVisualizer />} />
        </Routes>
      </Router>
    </ErrorBoundary>
  );
}

export default App;
