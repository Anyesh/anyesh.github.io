import { site } from "../config.js";

export default function Hero() {
  return (
    <section className="hero">
      <p className="hero-kicker">{site.kicker}</p>
      <h1 className="hero-title">{site.tagline}</h1>
      <p className="hero-intro">{site.intro}</p>
    </section>
  );
}
