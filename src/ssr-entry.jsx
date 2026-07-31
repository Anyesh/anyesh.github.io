import { renderToString } from "react-dom/server";
import { StaticRouter } from "react-router-dom/server";
import { loaders } from "virtual:artifact-loaders";
import Layout from "./components/Layout.jsx";
import ArtifactShell from "./components/ArtifactShell.jsx";
import GalleryGrid from "./components/GalleryGrid.jsx";
import Hero from "./components/Hero.jsx";

// main.jsx mounts with createRoot, not hydrateRoot, so React discards this markup
// and re-renders on mount. That is deliberate: several artifacts seed state with
// Math.random and would fail hydration. Crawlers already have the prose by then.
export async function renderArticle(meta) {
  const loader = loaders[meta.slug];
  if (!loader) throw new Error(`no loader for slug "${meta.slug}"`);
  const { default: Artifact } = await loader();

  return renderToString(
    <StaticRouter location={`/a/${meta.slug}`}>
      <Layout>
        <ArtifactShell meta={meta}>
          <Artifact />
        </ArtifactShell>
      </Layout>
    </StaticRouter>
  );
}

// Rendered without Home.jsx's PAGE_SIZE cap so every artifact gets a crawlable
// inbound link; the client re-renders with search and category filters on mount.
export function renderHome(manifest) {
  return renderToString(
    <StaticRouter location="/">
      <Layout>
        <Hero />
        <GalleryGrid articles={manifest} />
      </Layout>
    </StaticRouter>
  );
}

// Falls back to header-only markup if an artifact ever throws during render.
export function renderArticleFallback(meta) {
  return renderToString(
    <StaticRouter location={`/a/${meta.slug}`}>
      <Layout>
        <ArtifactShell meta={meta}>
          <div className="canvas-loading">Loading artifact…</div>
        </ArtifactShell>
      </Layout>
    </StaticRouter>
  );
}
