import { Link } from "react-router-dom";

export default function NotFound() {
  return (
    <div className="empty">
      <p>That page wandered off.</p>
      <p className="empty-sub">
        <Link to="/" className="back">
          ← Back to the lab
        </Link>
      </p>
    </div>
  );
}
