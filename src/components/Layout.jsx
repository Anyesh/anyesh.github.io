import { Link } from "react-router-dom";
import { site } from "../config.js";

export default function Layout({ children }) {
  return (
    <div className="shell">
      <header className="masthead">
        <Link to="/" className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <span className="brand-text">
            <span className="brand-name">{site.name}</span>
            <span className="brand-kicker">{site.kicker}</span>
          </span>
        </Link>
        <nav className="masthead-links">
          {site.links.map((l) => (
            <a key={l.href} href={l.href} target="_blank" rel="noreferrer">
              {l.label}
            </a>
          ))}
        </nav>
      </header>

      <main className="main">{children}</main>

      <footer className="footer">
        <span>
          {site.name} — built by {site.author}
        </span>
        <nav className="footer-links">
          {site.links.map((l) => (
            <a key={l.href} href={l.href} target="_blank" rel="noreferrer">
              {l.label}
            </a>
          ))}
        </nav>
      </footer>
    </div>
  );
}
