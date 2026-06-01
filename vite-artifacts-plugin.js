import fs from "node:fs";
import path from "node:path";

const VIRTUAL_ID = "virtual:artifact-loaders";
const RESOLVED_ID = "\0" + VIRTUAL_ID;
const MANIFEST_FILE = "artifacts-manifest.json";

function toManifest(list) {
  return list.map(({ file, ...meta }) => meta);
}

function extractMetaObject(source) {
  const marker = source.indexOf("export const meta");
  if (marker === -1) return null;
  const open = source.indexOf("{", marker);
  if (open === -1) return null;

  let depth = 0;
  let quote = null;
  let escaped = false;
  for (let i = open; i < source.length; i++) {
    const ch = source[i];
    if (quote) {
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === quote) quote = null;
      continue;
    }
    if (ch === '"' || ch === "'" || ch === "`") {
      quote = ch;
      continue;
    }
    if (ch === "{") depth++;
    else if (ch === "}") {
      depth--;
      if (depth === 0) return source.slice(open, i + 1);
    }
  }
  return null;
}

const ARTIFACT_EXT = /\.(jsx|tsx)$/;

function readArtifacts(dir) {
  if (!fs.existsSync(dir)) return [];
  const seen = new Map();
  return fs
    .readdirSync(dir)
    .filter((f) => ARTIFACT_EXT.test(f))
    .map((file) => {
      const slugBase = file.replace(ARTIFACT_EXT, "");
      const source = fs.readFileSync(path.join(dir, file), "utf8");
      const objText = extractMetaObject(source);
      let meta = {};
      if (objText) {
        try {
          // meta is authored as a static literal, so evaluating it at build is safe.
          meta = new Function(`return (${objText});`)();
        } catch (err) {
          throw new Error(
            `Could not parse the meta object in artifacts/${file}: ${err.message}`
          );
        }
      }
      const slug = meta.slug || slugBase;
      // Two source files resolving to one slug would silently shadow each
      // other at the /a/:slug route, so fail loud instead.
      if (seen.has(slug)) {
        throw new Error(
          `Two artifacts claim the slug "${slug}": ${seen.get(slug)} and ${file}. ` +
            `Rename one or give it an explicit meta.slug.`
        );
      }
      seen.set(slug, file);
      return {
        slug,
        title: meta.title || slugBase,
        category: meta.category || "Uncategorized",
        description: meta.description || "",
        date: meta.date || "",
        tags: Array.isArray(meta.tags) ? meta.tags : [],
        file,
      };
    })
    .sort((a, b) => (b.date || "").localeCompare(a.date || ""));
}

export default function artifactsPlugin() {
  let artifactsDir;
  let server;

  function invalidate() {
    if (!server) return;
    const mod = server.moduleGraph.getModuleById(RESOLVED_ID);
    if (mod) {
      server.moduleGraph.invalidateModule(mod);
      server.ws.send({ type: "full-reload" });
    }
  }

  return {
    name: "learning-lab-artifacts",
    configResolved(config) {
      artifactsDir = path.resolve(config.root, "src/artifacts");
    },
    configureServer(s) {
      server = s;
      s.watcher.add(path.join(artifactsDir, "*.{jsx,tsx}"));
      const onChange = (file) => {
        if (file.startsWith(artifactsDir)) invalidate();
      };
      s.watcher.on("add", onChange);
      s.watcher.on("unlink", onChange);
      s.watcher.on("change", onChange);

      // Served live (re-read per request) so the gallery sees new artifacts without a restart.
      s.middlewares.use((req, res, next) => {
        const url = (req.url || "").split("?")[0];
        if (!url.endsWith(`/${MANIFEST_FILE}`)) return next();
        const body = JSON.stringify(toManifest(readArtifacts(artifactsDir)));
        res.setHeader("Content-Type", "application/json");
        res.setHeader("Cache-Control", "no-store");
        res.end(body);
      });
    },
    resolveId(id) {
      if (id === VIRTUAL_ID) return RESOLVED_ID;
    },
    load(id) {
      if (id !== RESOLVED_ID) return;
      const list = readArtifacts(artifactsDir);
      const entries = list
        .map((a) => `  ${JSON.stringify(a.slug)}: () => import("/src/artifacts/${a.file}")`)
        .join(",\n");
      return `export const loaders = {\n${entries}\n};\n`;
    },
    generateBundle() {
      const source = JSON.stringify(toManifest(readArtifacts(artifactsDir)));
      this.emitFile({ type: "asset", fileName: MANIFEST_FILE, source });
    },
  };
}
