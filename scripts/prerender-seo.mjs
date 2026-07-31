import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const BASE = "https://learn.anyesh.me";
const PORTFOLIO = "https://www.anyesh.me";
const DIST = "dist";
const SSR_DIST = "dist-ssr";
const SITE_NAME = "Learning Lab";
const AUTHOR = "Anish Shrestha";
const OG_IMAGE = `${BASE}/og-image.png`;
const SEO_START = "<!-- SEO:start -->";
const SEO_END = "<!-- SEO:end -->";
const BODY_START = "<!-- BODY:start -->";
const BODY_END = "<!-- BODY:end -->";

const escAttr = (s) =>
  String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

// JSON-LD is embedded in a script tag, so close-tag sequences in the data would
// break out of it; escape "<" to its unicode form to keep the block intact.
const jsonLd = (obj) => JSON.stringify(obj).replace(/</g, "\\u003c");

function artifactHead(a) {
  const url = `${BASE}/a/${a.slug}`;
  const title = `${a.title} — ${SITE_NAME}`;
  const desc = a.description || `${a.title}, an interactive explorable explanation.`;
  const graph = jsonLd({
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "TechArticle",
        headline: a.title,
        description: desc,
        url,
        datePublished: a.date || undefined,
        author: { "@type": "Person", name: AUTHOR, url: PORTFOLIO },
        publisher: { "@type": "Organization", name: SITE_NAME, url: `${BASE}/` },
        image: OG_IMAGE,
        articleSection: a.category || undefined,
        keywords: Array.isArray(a.tags) && a.tags.length ? a.tags.join(", ") : undefined,
        isPartOf: { "@type": "WebSite", name: SITE_NAME, url: `${BASE}/` },
        inLanguage: "en",
      },
      {
        "@type": "BreadcrumbList",
        itemListElement: [
          { "@type": "ListItem", position: 1, name: SITE_NAME, item: `${BASE}/` },
          { "@type": "ListItem", position: 2, name: a.category, item: `${BASE}/` },
          { "@type": "ListItem", position: 3, name: a.title, item: url },
        ],
      },
    ],
  });

  return [
    SEO_START,
    `    <title>${escAttr(title)}</title>`,
    `    <meta name="description" content="${escAttr(desc)}" />`,
    `    <link rel="canonical" href="${url}" />`,
    `    <meta name="author" content="${escAttr(AUTHOR)}" />`,
    `    <meta name="robots" content="index, follow, max-image-preview:large, max-snippet:-1" />`,
    `    <meta property="og:type" content="article" />`,
    `    <meta property="og:site_name" content="${SITE_NAME}" />`,
    `    <meta property="og:title" content="${escAttr(a.title)}" />`,
    `    <meta property="og:description" content="${escAttr(desc)}" />`,
    `    <meta property="og:url" content="${url}" />`,
    `    <meta property="og:image" content="${OG_IMAGE}" />`,
    a.date ? `    <meta property="article:published_time" content="${a.date}" />` : null,
    `    <meta property="article:author" content="${escAttr(AUTHOR)}" />`,
    `    <meta name="twitter:card" content="summary_large_image" />`,
    `    <meta name="twitter:title" content="${escAttr(a.title)}" />`,
    `    <meta name="twitter:description" content="${escAttr(desc)}" />`,
    `    <meta name="twitter:image" content="${OG_IMAGE}" />`,
    `    <script type="application/ld+json">${graph}</script>`,
    `    ${SEO_END}`,
  ]
    .filter(Boolean)
    .join("\n");
}

function homeHead(count) {
  const desc =
    `${count} interactive explorable explanations covering machine learning, transformers, ` +
    "maths, algorithms, and computer systems. Every one is a sandbox you can poke at.";
  const graph = jsonLd({
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "WebSite",
        "@id": `${BASE}/#website`,
        name: SITE_NAME,
        url: `${BASE}/`,
        description: desc,
        inLanguage: "en",
        author: { "@type": "Person", name: AUTHOR, url: PORTFOLIO },
      },
      {
        "@type": "CollectionPage",
        url: `${BASE}/`,
        name: `${SITE_NAME} — Interactive Explorable Explanations`,
        description: desc,
        isPartOf: { "@id": `${BASE}/#website` },
      },
    ],
  });

  return [
    SEO_START,
    `    <title>${SITE_NAME} — Interactive Explorable Explanations</title>`,
    `    <meta name="description" content="${escAttr(desc)}" />`,
    `    <link rel="canonical" href="${BASE}/" />`,
    `    <meta name="author" content="${escAttr(AUTHOR)}" />`,
    `    <meta name="robots" content="index, follow, max-image-preview:large, max-snippet:-1" />`,
    `    <meta property="og:type" content="website" />`,
    `    <meta property="og:site_name" content="${SITE_NAME}" />`,
    `    <meta property="og:title" content="${SITE_NAME} — Interactive Explorable Explanations" />`,
    `    <meta property="og:description" content="${escAttr(desc)}" />`,
    `    <meta property="og:url" content="${BASE}/" />`,
    `    <meta property="og:image" content="${OG_IMAGE}" />`,
    `    <meta name="twitter:card" content="summary_large_image" />`,
    `    <meta name="twitter:title" content="${SITE_NAME} — Interactive Explorable Explanations" />`,
    `    <meta name="twitter:description" content="${escAttr(desc)}" />`,
    `    <meta name="twitter:image" content="${OG_IMAGE}" />`,
    `    <script type="application/ld+json">${graph}</script>`,
    `    ${SEO_END}`,
  ].join("\n");
}

const template = readFileSync(join(DIST, "index.html"), "utf8");
for (const marker of [SEO_START, SEO_END, BODY_START, BODY_END]) {
  if (!template.includes(marker)) {
    throw new Error(`marker ${marker} not found in dist/index.html; cannot prerender`);
  }
}

const headBefore = template.slice(0, template.indexOf(SEO_START));
const headAfter = template.slice(template.indexOf(SEO_END) + SEO_END.length);

// The body markers sit inside headAfter; splitting again keeps one template.
const bodyBefore = headAfter.slice(0, headAfter.indexOf(BODY_START) + BODY_START.length);
const bodyAfter = headAfter.slice(headAfter.indexOf(BODY_END));

const page = (head, body) =>
  headBefore + head + bodyBefore + `\n    <div id="root">${body}</div>\n    ` + bodyAfter;

const manifest = JSON.parse(readFileSync(join(DIST, "artifacts-manifest.json"), "utf8"));

const ssrPath = join(process.cwd(), SSR_DIST, "ssr-entry.js");
if (!existsSync(ssrPath)) {
  throw new Error(
    `${ssrPath} missing. Run \`vite build --ssr src/ssr-entry.jsx --outDir ${SSR_DIST}\` first.`,
  );
}
const ssr = await import(pathToFileURL(ssrPath).href);

mkdirSync(join(DIST, "a"), { recursive: true });

let rendered = 0;
let fallbacks = 0;
for (const a of manifest) {
  let body;
  try {
    body = await ssr.renderArticle(a);
    rendered++;
  } catch (err) {
    // A future artifact touching a browser global in its render body must degrade
    // to header-only markup, not fail the whole deploy.
    console.warn(`[prerender] ${a.slug} failed to render: ${err.message}`);
    body = ssr.renderArticleFallback(a);
    fallbacks++;
  }
  writeFileSync(join(DIST, "a", `${a.slug}.html`), page(artifactHead(a), body));
}

writeFileSync(join(DIST, "index.html"), page(homeHead(manifest.length), ssr.renderHome(manifest)));

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
        `    <priority>${u.priority}</priority>\n  </url>`,
    )
    .join("\n") +
  `\n</urlset>\n`;
writeFileSync(join(DIST, "sitemap.xml"), sitemap);

const AI_AGENTS = [
  "GPTBot",
  "OAI-SearchBot",
  "ChatGPT-User",
  "ClaudeBot",
  "Claude-Web",
  "Claude-SearchBot",
  "anthropic-ai",
  "PerplexityBot",
  "Perplexity-User",
  "Google-Extended",
  "Applebot-Extended",
  "CCBot",
  "Bytespider",
  "cohere-ai",
  "Meta-ExternalAgent",
];
writeFileSync(
  join(DIST, "robots.txt"),
  `User-agent: *\nAllow: /\n\n` +
    `# Explicitly welcome AI/LLM crawlers and answer engines.\n` +
    AI_AGENTS.map((a) => `User-agent: ${a}\nAllow: /\n`).join("\n") +
    `\n# Structured summary for LLM consumers: ${BASE}/llms.txt\n` +
    `Sitemap: ${BASE}/sitemap.xml\n`,
);

const byCategory = new Map();
for (const a of manifest) {
  if (!byCategory.has(a.category)) byCategory.set(a.category, []);
  byCategory.get(a.category).push(a);
}
const llms = [
  `# ${SITE_NAME}`,
  "",
  `> ${manifest.length} interactive explorable explanations by ${AUTHOR}. Each one rebuilds a hard idea as a sandbox you can poke at.`,
  "",
  `Built and written by ${AUTHOR} (${PORTFOLIO}). Every article at ${BASE}/a/<slug> is a self-contained interactive artifact with a written explanation alongside it.`,
  "",
  ...[...byCategory.entries()]
    .sort((a, b) => a[0].localeCompare(b[0]))
    .flatMap(([category, items]) => [
      `## ${category}`,
      "",
      ...items.map((a) => `- [${a.title}](${BASE}/a/${a.slug})${a.date ? ` (${a.date})` : ""}: ${a.description}`),
      "",
    ]),
  "## Elsewhere",
  "",
  `- Portfolio and projects: ${PORTFOLIO}`,
  `- Sitemap: ${BASE}/sitemap.xml`,
  "",
].join("\n");
writeFileSync(join(DIST, "llms.txt"), llms);

console.log(
  `prerendered ${manifest.length} articles (${rendered} rendered, ${fallbacks} fallback), ` +
    `homepage with ${manifest.length} links, sitemap with ${urls.length} urls, robots.txt, llms.txt`,
);
