import { Routes, Route } from "react-router-dom";
import Layout from "./components/Layout.jsx";
import Home from "./components/Home.jsx";
import ArtifactView from "./components/ArtifactView.jsx";
import NotFound from "./components/NotFound.jsx";

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/a/:slug" element={<ArtifactView />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </Layout>
  );
}
