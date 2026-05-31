# Learning Lab

Interactive artifacts that make complicated ideas click. Each piece started as an
explanation (often something Claude sketched while teaching me a concept) and was
rebuilt into a sandbox you can poke at. Live at **old.anyesh.me**.

Built with Vite + React, deployed to GitHub Pages via Actions.

## Add an artifact

1. Drop a self-contained component at `src/artifacts/your-thing.jsx`.
2. Default-export the component, and add a `meta` export:

   ```jsx
   export const meta = {
     title: "What it teaches",
     category: "Bayesian Inference",
     description: "One or two sentences for the gallery card.",
     date: "2026-05-31",
     tags: ["mcmc", "uncertainty"],
   };

   export default function App() {
     return <div>...</div>;
   }
   ```

3. That's it. The registry auto-discovers it via `import.meta.glob`, it appears on
   the home grid, and it gets its own page at `/a/your-thing`. The URL slug comes
   from the filename unless you set `meta.slug`.

If an artifact pulls in a new library (charts, math, etc.), install it with
`npm install <pkg>` so the build can resolve the import.

## Develop

```bash
npm install
npm run dev      # local dev server
npm run build    # production build into dist/
npm run preview  # serve the built dist/ locally
```

## Deploy

Pushing to `master` triggers `.github/workflows/deploy.yml`, which builds and
publishes `dist/` to GitHub Pages. The custom domain (`public/CNAME`) and the SPA
404 fallback are handled in the build.

> One-time setup: in the repo's **Settings → Pages**, set **Source** to
> **GitHub Actions** (the old "Deploy from a branch" mode served the previous
> compiled site and must be switched off).
