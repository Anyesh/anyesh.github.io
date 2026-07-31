import { Link } from "react-router-dom";

// Shared by the client gallery and the build-time SSR pass so the prerendered
// homepage carries a real crawlable link to every artifact.
export default function GalleryGrid({ articles }) {
  return (
    <div className="grid">
      {articles.map((a, i) => (
        <Link
          key={a.slug}
          to={`/a/${a.slug}`}
          className="card"
          style={{ "--i": Math.min(i, 12) }}
        >
          <div className="card-top">
            <span className="card-cat">{a.category}</span>
            {a.date && <span className="card-date">{a.date}</span>}
          </div>
          <h2 className="card-title">{a.title}</h2>
          {a.description && <p className="card-desc">{a.description}</p>}
          {a.tags && a.tags.length > 0 && (
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
  );
}
