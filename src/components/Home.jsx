import { useState } from "react";
import { Link } from "react-router-dom";
import { artifacts, categories } from "../registry.js";
import { site } from "../config.js";

export default function Home() {
  const [active, setActive] = useState("all");
  const shown =
    active === "all" ? artifacts : artifacts.filter((a) => a.category === active);

  return (
    <>
      <section className="hero">
        <p className="hero-kicker">{site.kicker}</p>
        <h1 className="hero-title">{site.tagline}</h1>
        <p className="hero-intro">{site.intro}</p>
      </section>

      {artifacts.length === 0 ? (
        <div className="empty">
          <p>No artifacts yet.</p>
          <p className="empty-sub">
            Add one at <code>src/artifacts/your-thing.jsx</code> with a{" "}
            <code>meta</code> export and a default component.
          </p>
        </div>
      ) : (
        <>
          {categories.length > 1 && (
            <div className="filters" role="tablist" aria-label="Categories">
              <button
                className={`chip ${active === "all" ? "chip-on" : ""}`}
                onClick={() => setActive("all")}
              >
                All <span className="chip-count">{artifacts.length}</span>
              </button>
              {categories.map((c) => {
                const n = artifacts.filter((a) => a.category === c).length;
                return (
                  <button
                    key={c}
                    className={`chip ${active === c ? "chip-on" : ""}`}
                    onClick={() => setActive(c)}
                  >
                    {c} <span className="chip-count">{n}</span>
                  </button>
                );
              })}
            </div>
          )}

          <div className="grid">
            {shown.map((a, i) => (
              <Link
                key={a.slug}
                to={`/a/${a.slug}`}
                className="card"
                style={{ "--i": i }}
              >
                <div className="card-top">
                  <span className="card-cat">{a.category}</span>
                  {a.date && <span className="card-date">{a.date}</span>}
                </div>
                <h2 className="card-title">{a.title}</h2>
                {a.description && (
                  <p className="card-desc">{a.description}</p>
                )}
                {a.tags.length > 0 && (
                  <div className="card-tags">
                    {a.tags.map((t) => (
                      <span key={t} className="tag">
                        {t}
                      </span>
                    ))}
                  </div>
                )}
                <span className="card-go">Open →</span>
              </Link>
            ))}
          </div>
        </>
      )}
    </>
  );
}
