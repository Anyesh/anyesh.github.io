import { Suspense, lazy } from "react";
import { Routes, Route } from "react-router-dom";
import Layout from "./components/Layout.jsx";
import NotFound from "./components/NotFound.jsx";

const Home = lazy(() => import("./components/Home.jsx"));
const ArtifactView = lazy(() => import("./components/ArtifactView.jsx"));

export default function App() {
  return (
    <Layout>
      <Suspense fallback={<div className="canvas-loading">Loading…</div>}>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/a/:slug" element={<ArtifactView />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </Suspense>
    </Layout>
  );
}
