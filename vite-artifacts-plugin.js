import fs from "node:fs";
import path from "node:path";

const VIRTUAL_ID = "virtual:artifacts";
const RESOLVED_ID = "\0" + VIRTUAL_ID;

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

function readArtifacts(dir) {
  if (!fs.existsSync(dir)) return [];
  return fs
    .readdirSync(dir)
    .filter((f) => f.endsWith(".jsx"))
    .map((file) => {
      const slugBase = file.replace(/\.jsx$/, "");
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
      return {
        slug: meta.slug || slugBase,
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
      s.watcher.add(path.join(artifactsDir, "*.jsx"));
      const onChange = (file) => {
        if (file.startsWith(artifactsDir)) invalidate();
      };
      s.watcher.on("add", onChange);
      s.watcher.on("unlink", onChange);
      s.watcher.on("change", onChange);
    },
    resolveId(id) {
      if (id === VIRTUAL_ID) return RESOLVED_ID;
    },
    load(id) {
      if (id !== RESOLVED_ID) return;
      const list = readArtifacts(artifactsDir);
      const entries = list
        .map((a) => {
          const meta = { ...a };
          delete meta.file;
          return `  { ...${JSON.stringify(meta)}, load: () => import("/src/artifacts/${a.file}") }`;
        })
        .join(",\n");
      return `export const artifacts = [\n${entries}\n];\n`;
    },
  };
}
