import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { join } from "node:path";

const BASE = "https://learn.anyesh.me";
const DIST = "dist";
const SITE_NAME = "Learning Lab";
const AUTHOR = "Anish Shrestha";
const OG_IMAGE = `${BASE}/og-image.png`;
const START = "<!-- SEO:start -->";
const END = "<!-- SEO:end -->";

const escAttr = (s) =>
  String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

// JSON-LD is embedded in a script tag, so close-tag sequences in the data would
// break out of it; escape "<" to its unicode form to keep the block intact.
const jsonLd = (obj) =>
  JSON.stringify(obj).replace(/</g, "\\u003c");

function artifactBlock(a) {
  const url = `${BASE}/a/${a.slug}`;
  const title = `${a.title} — ${SITE_NAME}`;
  const desc = a.description || `${a.title}, an interactive explorable explanation.`;
  const ld = jsonLd({
    "@context": "https://schema.org",
    "@type": "TechArticle",
    headline: a.title,
    description: desc,
    url,
    datePublished: a.date || undefined,
    author: { "@type": "Person", name: AUTHOR },
    publisher: { "@type": "Organization", name: SITE_NAME },
    image: OG_IMAGE,
    articleSection: a.category || undefined,
    keywords: Array.isArray(a.tags) && a.tags.length ? a.tags.join(", ") : undefined,
    isPartOf: { "@type": "WebSite", name: SITE_NAME, url: `${BASE}/` },
  });
  return [
    START,
    `    <title>${escAttr(title)}</title>`,
    `    <meta name="description" content="${escAttr(desc)}" />`,
    `    <link rel="canonical" href="${url}" />`,
    `    <meta property="og:type" content="article" />`,
    `    <meta property="og:site_name" content="${SITE_NAME}" />`,
    `    <meta property="og:title" content="${escAttr(a.title)}" />`,
    `    <meta property="og:description" content="${escAttr(desc)}" />`,
    `    <meta property="og:url" content="${url}" />`,
    `    <meta property="og:image" content="${OG_IMAGE}" />`,
    `    <meta name="twitter:card" content="summary_large_image" />`,
    `    <meta name="twitter:title" content="${escAttr(a.title)}" />`,
    `    <meta name="twitter:description" content="${escAttr(desc)}" />`,
    `    <meta name="twitter:image" content="${OG_IMAGE}" />`,
    `    <script type="application/ld+json">${ld}</script>`,
    `    ${END}`,
  ].join("\n");
}

const template = readFileSync(join(DIST, "index.html"), "utf8");
if (!template.includes(START) || !template.includes(END)) {
  throw new Error("SEO markers not found in dist/index.html; cannot prerender heads.");
}

const manifest = JSON.parse(readFileSync(join(DIST, "artifacts-manifest.json"), "utf8"));

const before = template.slice(0, template.indexOf(START));
const after = template.slice(template.indexOf(END) + END.length);

mkdirSync(join(DIST, "a"), { recursive: true });
for (const a of manifest) {
  const html = before + artifactBlock(a) + after;
  writeFileSync(join(DIST, "a", `${a.slug}.html`), html);
}

const dates = manifest.map((a) => a.date).filter(Boolean).sort();
const homeLastmod = dates.length ? dates[dates.length - 1] : "";
const urls = [
  { loc: `${BASE}/`, lastmod: homeLastmod, priority: "1.0" },
  ...manifest.map((a) => ({ loc: `${BASE}/a/${a.slug}`, lastmod: a.date || "", priority: "0.8" })),
];
const sitemap =
  `<?xml version="1.0" encoding="UTF-8"?>\n` +
  `<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n` +
  urls
    .map(
      (u) =>
        `  <url>\n    <loc>${u.loc}</loc>\n` +
        (u.lastmod ? `    <lastmod>${u.lastmod}</lastmod>\n` : "") +
        `    <priority>${u.priority}</priority>\n  </url>`
    )
    .join("\n") +
  `\n</urlset>\n`;
writeFileSync(join(DIST, "sitemap.xml"), sitemap);

if (!existsSync(join(DIST, "robots.txt"))) {
  writeFileSync(
    join(DIST, "robots.txt"),
    `User-agent: *\nAllow: /\n\nSitemap: ${BASE}/sitemap.xml\n`
  );
}

console.log(`prerendered ${manifest.length} artifact heads, sitemap with ${urls.length} urls`);
