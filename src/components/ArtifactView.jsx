import { Suspense, lazy, useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { loaders } from "virtual:artifact-loaders";
import { getManifest } from "../registry.js";
import ArtifactShell from "./ArtifactShell.jsx";
import NotFound from "./NotFound.jsx";

export default function ArtifactView() {
  const { slug } = useParams();
  const [manifest, setManifest] = useState(null);

  useEffect(() => {
    let alive = true;
    getManifest()
      .then((m) => alive && setManifest(m))
      .catch(() => alive && setManifest([]));
    return () => {
      alive = false;
    };
  }, []);

  const loader = loaders[slug];
  const meta = manifest ? manifest.find((a) => a.slug === slug) : null;

  useEffect(() => {
    if (meta) document.title = `${meta.title} — Learning Lab`;
    return () => {
      document.title = "Learning Lab";
    };
  }, [meta]);

  const Component = useMemo(() => (loader ? lazy(loader) : null), [loader]);

  if (!loader) return <NotFound />;

  return (
    <ArtifactShell meta={meta}>
      <Suspense fallback={<div className="canvas-loading">Loading artifact…</div>}>
        <Component />
      </Suspense>
    </ArtifactShell>
  );
}
