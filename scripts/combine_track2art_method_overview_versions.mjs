import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const ROOT = process.cwd();
const OUT_DIR = path.join(ROOT, "outputs");
const INPUT = path.join(OUT_DIR, "track2art_method_overview.pptx");
const REFINED = path.join(OUT_DIR, "track2art_method_overview_v2_refined.pptx");
const ORIGINAL_IMAGE = "/Users/lxt/Downloads/image.png";

async function bytes(filePath) {
  const b = await fs.readFile(filePath);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}

function addVersionLabel(slide, value, fill, stroke, color) {
  const pill = slide.shapes.add({
    geometry: "roundRect",
    name: `version_${value.replaceAll(" ", "_")}`,
    position: { left: 1129, top: 682, width: 136, height: 27 },
    fill,
    line: { style: "solid", fill: stroke, width: 1 },
    borderRadius: "rounded-md",
  });
  pill.text = value;
  pill.text.style = {
    fontFamily: "Arial", fontSize: 10.5, bold: true, color, alignment: "center",
  };
  pill.text.verticalAlignment = "middle";
}

async function main() {
  await fs.copyFile(INPUT, REFINED);
  const presentation = await PresentationFile.importPptx(await FileBlob.load(INPUT));
  const refinedSlide = presentation.slides.getItem(0);
  const v2 = refinedSlide.duplicate();
  v2.moveTo(1);

  refinedSlide.images.add({
    blob: await bytes(ORIGINAL_IMAGE),
    contentType: "image/png",
    alt: "Track2Art method overview, original version",
    name: "v1_original_render",
    fit: "contain",
    position: { left: 0, top: 0, width: 1280, height: 720 },
  });
  addVersionLabel(refinedSlide, "V1 · Original", "FFFFFFE8", "6DAEE0", "153D74");
  addVersionLabel(v2, "V2 · Refined", "FFFFFFE8", "90B86C", "195C25");

  const v1Preview = await presentation.export({ slide: refinedSlide, format: "png", scale: 1.5 });
  await fs.writeFile(path.join(OUT_DIR, "track2art_method_overview_v1_original_preview.png"), new Uint8Array(await v1Preview.arrayBuffer()));
  const v2Preview = await presentation.export({ slide: v2, format: "png", scale: 1.5 });
  await fs.writeFile(path.join(OUT_DIR, "track2art_method_overview_v2_refined_preview.png"), new Uint8Array(await v2Preview.arrayBuffer()));

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(INPUT);
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
