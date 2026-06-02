# Learning Lab

Interactive artifacts that make hard ideas click. Each one runs the real
algorithm in your browser (no faked numbers) and lets you poke at it: machine
learning, maths, algorithms, systems, signals, and the odd corner of computer
science. Live at **[anyesh.github.io](https://anyesh.github.io)**.

Built with Vite + React.

## Contributing

New artifacts are welcome. The model is drop-in: add a self-contained component
at `src/artifacts/your-thing.jsx` with a default export and a `meta` export, and
it auto-registers into the gallery with its own page at `/a/your-thing`.

```jsx
export const meta = {
  title: "What it teaches",
  category: "Linear Algebra",
  description: "One or two sentences for the gallery card.",
  date: "2026-06-02",
  tags: ["eigenvectors", "geometry"],
};

export default function App() {
  return <div>...</div>;
}
```

Run `npm install` then `npm run dev` to work on it locally, and `npm run build`
to check it compiles. Open a pull request when it works.

## Support

If these are useful to you, you can
[buy me a coffee](https://buymeacoffee.com/anyesh).

## License

[MIT](LICENSE).
