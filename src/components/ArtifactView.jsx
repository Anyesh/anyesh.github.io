import { Suspense, lazy, useEffect, useMemo } from "react";
import { Link, useParams } from "react-router-dom";
import { getArtifact } from "../registry.js";
import NotFound from "./NotFound.jsx";

export default function ArtifactView() {
  const { slug } = useParams();
  const artifact = getArtifact(slug);

  useEffect(() => {
    if (artifact) document.title = `${artifact.title} — Learning Lab`;
    return () => {
      document.title = "Learning Lab";
    };
  }, [artifact]);

  const Component = useMemo(
    () => (artifact ? lazy(artifact.load) : null),
    [artifact]
  );

  if (!artifact) return <NotFound />;

  return (
    <article className="artifact">
      <Link to="/" className="back">
        ← All artifacts
      </Link>

      <header className="artifact-head">
        <span className="artifact-cat">{artifact.category}</span>
        <h1 className="artifact-title">{artifact.title}</h1>
        {artifact.description && (
          <p className="artifact-desc">{artifact.description}</p>
        )}
        {artifact.tags.length > 0 && (
          <div className="card-tags">
            {artifact.tags.map((t) => (
              <span key={t} className="tag">
                {t}
              </span>
            ))}
          </div>
        )}
      </header>

      <div className="canvas">
        <div className="canvas-bar">
          <span className="canvas-dot" aria-hidden="true" />
          <span className="canvas-bar-label">Live</span>
          <span className="canvas-bar-sep" aria-hidden="true">
            /
          </span>
          <span>Interactive · drag, toggle, run it</span>
        </div>
        <Suspense
          fallback={<div className="canvas-loading">Loading artifact…</div>}
        >
          <Component />
        </Suspense>
      </div>
    </article>
  );
}
