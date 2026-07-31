import { useState, useEffect, useMemo } from "react";
import { getManifest, categoriesOf } from "../registry.js";
import GalleryGrid from "./GalleryGrid.jsx";
import Hero from "./Hero.jsx";

const PAGE_SIZE = 48;

function matches(artifact, q) {
  if (!q) return true;
  const hay = [
    artifact.title,
    artifact.description,
    artifact.category,
    ...(artifact.tags || []),
  ]
    .join(" ")
    .toLowerCase();
  return q.split(/\s+/).every((term) => hay.includes(term));
}

export default function Home() {
  const [manifest, setManifest] = useState(null);
  const [error, setError] = useState(null);
  const [active, setActive] = useState("all");
  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(PAGE_SIZE);

  useEffect(() => {
    let alive = true;
    getManifest()
      .then((m) => alive && setManifest(m))
      .catch((e) => alive && setError(e));
    return () => {
      alive = false;
    };
  }, []);

  const categories = useMemo(() => (manifest ? categoriesOf(manifest) : []), [manifest]);

  const q = query.trim().toLowerCase();
  const filtered = useMemo(() => {
    if (!manifest) return [];
    return manifest.filter(
      (a) => (active === "all" || a.category === active) && matches(a, q)
    );
  }, [manifest, active, q]);

  useEffect(() => {
    setLimit(PAGE_SIZE);
  }, [active, q]);

  const shown = filtered.slice(0, limit);

  if (error) {
    return (
      <div className="empty">
        <p>Could not load the gallery.</p>
        <p className="empty-sub">{String(error.message || error)}</p>
      </div>
    );
  }

  return (
    <>
      <Hero />

      {manifest === null ? (
        <div className="gallery-loading">Loading the collection…</div>
      ) : manifest.length === 0 ? (
        <div className="empty">
          <p>No artifacts yet.</p>
          <p className="empty-sub">
            Add one at <code>src/artifacts/your-thing.jsx</code> with a <code>meta</code> export
            and a default component.
          </p>
        </div>
      ) : (
        <>
          <div className="toolbar">
            <input
              type="search"
              className="search"
              placeholder="Search artifacts…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              aria-label="Search artifacts by title, description, or tag"
            />
            {categories.length > 1 && (
              <div className="filters" role="tablist" aria-label="Categories">
                <button
                  className={`chip ${active === "all" ? "chip-on" : ""}`}
                  onClick={() => setActive("all")}
                >
                  All <span className="chip-count">{manifest.length}</span>
                </button>
                {categories.map(([c, n]) => (
                  <button
                    key={c}
                    className={`chip ${active === c ? "chip-on" : ""}`}
                    onClick={() => setActive(c)}
                  >
                    {c} <span className="chip-count">{n}</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          {filtered.length === 0 ? (
            <div className="empty">
              <p>Nothing matches that.</p>
              <p className="empty-sub">Try a different search or category.</p>
            </div>
          ) : (
            <>
              <GalleryGrid articles={shown} />

              <div className="gallery-foot">
                <span className="result-count">
                  Showing {shown.length} of {filtered.length}
                </span>
                {limit < filtered.length && (
                  <button className="load-more" onClick={() => setLimit((n) => n + PAGE_SIZE)}>
                    Load more
                  </button>
                )}
              </div>
            </>
          )}
        </>
      )}
    </>
  );
}
