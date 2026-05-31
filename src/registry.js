import { artifacts } from "virtual:artifacts";

export { artifacts };

export const categories = [...new Set(artifacts.map((a) => a.category))].sort();

export function getArtifact(slug) {
  return artifacts.find((a) => a.slug === slug);
}
