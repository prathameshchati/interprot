import "./App.css";
import { createHashRouter, RouterProvider } from "react-router-dom";
import LandingPage from "./components/LandingPage";
import SAEVisualizer from "./SAEVisualizer";
import ErrorBoundary from "./components/ErrorBoundary";

const router = createHashRouter([
  {
    path: "/",
    element: <LandingPage />,
  },
  {
    path: "/:model/:feature?",
    element: <SAEVisualizer />,
  },
]);

function App() {
  return (
    <ErrorBoundary>
      <RouterProvider router={router} />
    </ErrorBoundary>
  );
}

export default App;
