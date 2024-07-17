import assert from "node:assert";
import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { rimraf } from "rimraf";

const DATASET = process.argv[2];
const DATA_TYPE = process.argv[3];

assert(
  DATASET,
  "Must set dataset, usage: node convert.mjs <dataset> <segment|bbox>"
);

assert(
  DATA_TYPE === "segment" || DATA_TYPE === "bbox",
  "Must set segment or bbox, usage: node convert.mjs <dataset> <segment|bbox>"
);

const SRC_DIR = resolve("datasets", DATASET);

await rimraf(resolve(SRC_DIR, "labels"));
await mkdir(resolve(SRC_DIR, "labels"));

const files = await readdir(resolve(SRC_DIR, "images"));

const LABELS = new Set();
for (const fn of files.filter((fn) => fn.endsWith(".json"))) {
  const f = await readFile(resolve(SRC_DIR, "images", fn));
  const data = JSON.parse(f.toString());

  for (const shape of data.shapes) {
    LABELS.add(shape.label);
  }
}

const IDs = {};
[...LABELS].forEach((label, index) => (IDs[label] = index));

for (const fn of files.filter((fn) => fn.endsWith(".json"))) {
  const f = await readFile(resolve(SRC_DIR, "images", fn));
  const data = JSON.parse(f.toString());

  const bb = data.shapes
    .filter((shape) => shape.shape_type === "rectangle")
    .map(
      (shape) =>
        IDs[shape.label] +
        " " +
        shape.points
          .map(([x, y]) => x / data.imageWidth + " " + y / data.imageHeight)
          .join(" ")
    );

  const seg = data.shapes
    .filter((shape) => shape.shape_type === "polygon")
    .map(
      (shape) =>
        IDs[shape.label] +
        " " +
        shape.points
          .map(([x, y]) => x / data.imageWidth + " " + y / data.imageHeight)
          .join(" ")
    );

  if (bb.length && DATA_TYPE === "bbox") {
    await writeFile(
      resolve(SRC_DIR, "labels", fn.substring(0, fn.length - 5) + ".txt"),
      bb.join("\n")
    );
  }

  if (seg.length && DATA_TYPE === "segment") {
    await writeFile(
      resolve(SRC_DIR, "labels", fn.substring(0, fn.length - 5) + ".txt"),
      seg.join("\n")
    );
  }
}

writeFile(
  resolve(SRC_DIR, "config.yml"),
  `path: ${SRC_DIR}
train: ${SRC_DIR}/images
val: ${SRC_DIR}/images
test:

names:
${[...LABELS].map((label, index) => `  ${index}: ${label}`).join("\n")}`
);
