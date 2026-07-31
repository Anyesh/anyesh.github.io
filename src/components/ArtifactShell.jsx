import { Link } from "react-router-dom";

// Presentational only, so the build-time SSR pass and the client render emit the
// same markup. meta arrives as a prop rather than from getManifest() because the
// server has no fetch target and the h1/description must land in static HTML.
export default function ArtifactShell({ meta, children }) {
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
        {children}
      </div>
    </article>
  );
}
