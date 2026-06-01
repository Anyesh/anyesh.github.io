import { Suspense, lazy, useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { loaders } from "virtual:artifact-loaders";
import { getManifest } from "../registry.js";
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
    <article className="artifact">
      <Link to="/" className="back">
        ← All artifacts
      </Link>

      {meta && (
        <header className="artifact-head">
          <span className="artifact-cat">{meta.category}</span>
          <h1 className="artifact-title">{meta.title}</h1>
          {meta.description && <p className="artifact-desc">{meta.description}</p>}
          {meta.tags && meta.tags.length > 0 && (
            <div className="card-tags">
              {meta.tags.map((t) => (
                <span key={t} className="tag">
                  {t}
                </span>
              ))}
            </div>
          )}
        </header>
      )}

      <div className="canvas">
        <div className="canvas-bar">
          <span className="canvas-dot" aria-hidden="true" />
          <span className="canvas-bar-label">Live</span>
          <span className="canvas-bar-sep" aria-hidden="true">
            /
          </span>
          <span>Interactive · drag, toggle, run it</span>
        </div>
        <Suspense fallback={<div className="canvas-loading">Loading artifact…</div>}>
          <Component />
        </Suspense>
      </div>
    </article>
  );
}
