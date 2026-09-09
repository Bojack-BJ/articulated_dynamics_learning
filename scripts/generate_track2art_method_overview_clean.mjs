import fs from "node:fs/promises";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const ROOT = "/Users/lxt/Documents/New project";
const ASSET = path.join(ROOT, "paper_assets/refrigerator045");
const OUT = path.join(ROOT, "outputs");

const C = {
  ink: "#172033", muted: "#5A6474", white: "#FFFFFF", line: "#B8C4D0",
  stage1: "#EEF7FD", stage1b: "#9BCDF2", stage1l: "#6DAEE0",
  stage2: "#FBF2FD", stage2b: "#E7B7F3", stage2l: "#B879CE",
  stage3: "#F4FAEE", stage3b: "#CAE8A8", stage3l: "#90B86C",
  base: "#8A8A8A", door: "#F28E2B", drawer: "#397BC5", slot1: "#8CC66A", slot2: "#A965D8",
  blue: "#295EA8", paleBlue: "#E6F0FA", orange: "#DE7923", red: "#D94A3D",
};

async function bytes(p) {
  const b = await fs.readFile(p);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}

function addText(slide, text, x, y, w, h, opts = {}) {
  const s = slide.shapes.add({
    geometry: "textbox",
    name: opts.name,
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  s.text = text;
  s.text.style = {
    typeface: "Arial",
    fontSize: opts.size ?? 12,
    bold: opts.bold ?? false,
    italic: opts.italic ?? false,
    color: opts.color ?? C.ink,
    alignment: opts.align ?? "left",
    verticalAlignment: opts.valign ?? "middle",
    autoFit: "shrinkText",
    wrap: "square",
    insets: { left: opts.pad ?? 2, right: opts.pad ?? 2, top: 0, bottom: 0 },
  };
  return s;
}

function box(slide, x, y, w, h, fill, line, radius = true, name) {
  return slide.shapes.add({
    geometry: radius ? "roundRect" : "rect", name,
    position: { left: x, top: y, width: w, height: h },
    fill, line: { style: "solid", fill: line, width: 1 },
    ...(radius ? { borderRadius: "rounded-md" } : {}),
  });
}

function labelBox(slide, text, x, y, w, h, fill, line, opts = {}) {
  const s = box(slide, x, y, w, h, fill, line, opts.radius !== false, opts.name);
  s.text = text;
  s.text.style = {
    typeface: "Arial", fontSize: opts.size ?? 12, bold: opts.bold ?? true,
    color: opts.color ?? C.ink, alignment: opts.align ?? "center",
    verticalAlignment: "middle", autoFit: "shrinkText", wrap: "square",
    insets: { left: 4, right: 4, top: 2, bottom: 2 },
  };
  return s;
}

function arrow(slide, a, b, color = C.muted, kind = "straight", fromSide = "bottom", toSide = "top", dashed = false) {
  const connector = slide.shapes.connect(a, b, {
    kind, fromSide, toSide,
    line: { style: dashed ? "dashed" : "solid", fill: color, width: 1.4 },
    tail: { type: "triangle", width: "sm", length: "sm" },
  });
  connector.bringToFront();
  return connector;
}

async function addImage(slide, filename, x, y, w, h, name, fit = "contain", crop) {
  const img = slide.images.add({
    blob: await bytes(path.join(ASSET, filename)), contentType: "image/png",
    alt: `${name}: ${filename}`, fit,
    position: { left: x, top: y, width: w, height: h },
    ...(crop ? { crop } : {}),
  });
  img.name = name;
  return img;
}

function tokenRow(slide, x, y, n, size, colors, gap = 4, name = "tokens") {
  const arr = [];
  for (let i = 0; i < n; i++) {
    arr.push(slide.shapes.add({
      geometry: "rect", name: `${name}-${i + 1}`,
      position: { left: x + i * (size + gap), top: y, width: size, height: size },
      fill: colors[i % colors.length], line: { style: "solid", fill: "#5F7185", width: 0.7 },
    }));
  }
  return arr;
}

function stage(slide, x, w, title, fill, banner, border) {
  box(slide, x, 8, w, 704, fill, border, true, `${title}-panel`);
  const b = box(slide, x, 8, w, 38, banner, border, true, `${title}-banner`);
  addText(slide, title, x + 4, 9, w - 8, 36, { size: 20, bold: true, align: "center" });
  return b;
}

function miniHeatmap(slide, x, y, cols = 7, rows = 6, cw = 18, ch = 12) {
  const palette = ["#EDF2FA", "#D8E4F5", "#B6CBE8", "#7195CB", "#234D93"];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const center = Math.floor((r / rows) * cols);
      const d = Math.abs(c - center);
      const idx = d === 0 ? 4 : d === 1 ? 2 : d === 2 ? 1 : 0;
      box(slide, x + c * cw, y + r * ch, cw - 1, ch - 1, palette[idx], "#D7DFEA", false, `assignment_heatmap-r${r}c${c}`);
    }
  }
}

function pairMatrix(slide, x, y, n = 6, cell = 18) {
  const slotColors = [C.drawer, C.slot1, C.base, C.door, C.slot2, "#C9CDD2"];
  tokenRow(slide, x + cell, y - 20, n, 13, slotColors, 5, "pair-slot-top");
  tokenRow(slide, x - 20, y + 2, n, 13, slotColors, 5, "pair-slot-left");
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    const v = r === c ? "#C89FDC" : Math.abs(r - c) === 1 ? "#E9D8F1" : "#F6EFF9";
    box(slide, x + c * cell, y + r * cell, cell - 1, cell - 1, v, "#E3D3EA", false, `pair-feature-${r}-${c}`);
  }
}

function node(slide, text, x, y, w, h, fill, border, name) {
  return labelBox(slide, text, x, y, w, h, fill, border, { size: 11, bold: true, name });
}

async function main() {
  await fs.mkdir(OUT, { recursive: true });
  const p = Presentation.create({ slideSize: { width: 1280, height: 720 } });
  const s = p.slides.add();
  s.background.fill = C.white;

  const x1 = 8, w1 = 632, x2 = 644, w2 = 266, x3 = 914, w3 = 358;
  stage(s, x1, w1, "1. Motion-aware Part Discovery", C.stage1, C.stage1b, C.stage1l);
  stage(s, x2, w2, "2. Kinematic Reasoning", C.stage2, C.stage2b, C.stage2l);
  stage(s, x3, w3, "3. Geometric Joint Recovery", C.stage3, C.stage3b, C.stage3l);

  // Stage I left: actual RGB sequence, depth-like observation row, tracking, trajectories.
  const lx = 19, lw = 214;
  box(s, lx, 54, lw, 157, "#FFFFFF", C.stage1l, true, "rgbd-interaction-panel");
  addText(s, "RGB-D Interaction", lx + 8, 55, lw - 16, 22, { size: 14, bold: true, align: "center" });
  const rgbFiles = ["rgb_t0.png", "rgb_t1.png", "rgb_t2.png", "rgb_t3.png"];
  for (let i = 0; i < 4; i++) {
    const ix = lx + 9 + i * 49;
    await addImage(s, rgbFiles[i], ix, 79, 46, 49, `rgb-frame-${i}`, "cover");
    for (let k = 0; k < 4; k++) {
      box(s, ix + k * 12, 74, 8, 4, "#101820", "#101820", false);
      box(s, ix + k * 12, 129, 8, 4, "#101820", "#101820", false);
    }
  }
  addText(s, "Depth", lx + 8, 134, lw - 16, 15, { size: 10, bold: true, align: "center" });
  for (let i = 0; i < 4; i++) {
    const dx = lx + 9 + i * 49;
    await addImage(s, rgbFiles[i], dx, 150, 46, 49, `depth-source-${i}`, "cover");
    const overlay = box(s, dx, 150, 46, 49, { color: "#214E9A", transparency: 35 }, "none", false, `depth-overlay-${i}`);
    overlay.opacity = 0.75;
  }
  const trackBox = labelBox(s, "❄  Persistent Point Tracking\n(Frozen CoTracker3)", lx + 8, 220, lw - 16, 45, C.paleBlue, C.stage1l, { size: 12, name: "persistent-point-tracking" });
  const trajBox = box(s, lx + 8, 281, lw - 16, 414, "#FFFFFF", C.stage1l, true, "trajectory-panel");
  addText(s, "Persistent 4D Point Trajectories", lx + 15, 286, lw - 30, 25, { size: 13, bold: true, align: "center" });
  await addImage(s, "tracks_4d_clean.png", lx + 12, 313, lw - 24, 367, "trajectories_4d", "contain", { left: 0.28, top: 0.10, right: 0.22, bottom: 0.08 });
  addText(s, "persistent point identity across time", lx + 16, 669, lw - 32, 18, { size: 9, color: C.muted, align: "center" });
  arrow(s, trackBox, trajBox, C.stage1l);

  // Stage I right: editable token/slot architecture.
  const rx = 244, rw = 383;
  box(s, rx, 54, rw, 643, "#FFFFFF", C.stage1l, true, "token-slot-architecture");
  addText(s, "Per-track Token Construction", rx + 8, 55, rw - 16, 24, { size: 14, bold: true, align: "center", color: "#173A75" });
  const app = labelBox(s, "Track Appearance Feature\n(384D)", rx + 18, 87, 154, 54, "#F5F8FC", "#7FA9D5", { size: 11, name: "appearance-feature" });
  const geo = labelBox(s, "Trajectory Geometry Feature\n(32D)", rx + 211, 87, 154, 54, "#F7FBF4", "#92B678", { size: 11, name: "geometry-feature" });
  addText(s, "+", rx + 176, 94, 31, 38, { size: 23, bold: true, align: "center", color: C.muted });
  tokenRow(s, rx + 117, 158, 9, 18, ["#EDF3FA"], 5, "per-track-token");
  addText(s, "Per-track Token  (416D)", rx + 88, 181, 210, 18, { size: 10, bold: true, align: "center" });
  const enc = labelBox(s, "Transformer Encoder", rx + 38, 215, rw - 76, 40, "#DCEAF8", "#4B82BD", { size: 14, name: "transformer-encoder" });
  tokenRow(s, rx + 58, 267, 11, 17, ["#29486E", "#416488", "#5D7898"], 5, "encoded-track-token");
  addText(s, "Encoded track tokens", rx + 42, 288, 142, 15, { size: 9, color: C.muted, align: "center" });
  const queryColors = [C.drawer, C.slot1, C.base, C.door, C.slot2, "#C9CDD2"];
  tokenRow(s, rx + 230, 267, 6, 17, queryColors, 7, "learnable-part-query");
  addText(s, "Learnable Part Queries (K)", rx + 207, 288, 168, 15, { size: 9, bold: true, align: "center" });
  const dec = labelBox(s, "Transformer Decoder", rx + 38, 329, rw - 76, 43, "#DCEAF8", "#4B82BD", { size: 14, name: "transformer-decoder" });
  arrow(s, enc, dec, "#4B82BD", "straight");
  arrow(s, geo, enc, "#7F9B72", "straight");
  arrow(s, app, enc, "#4B82BD", "straight");
  const qAnchor = box(s, rx + 222, 264, 151, 24, "none", "none", false, "query-anchor");
  arrow(s, qAnchor, dec, C.slot2, "elbow", "bottom", "right", true);
  addText(s, "Cross-attention", rx + 292, 303, 82, 18, { size: 9, color: C.slot2, align: "center" });
  tokenRow(s, rx + 119, 390, 6, 22, queryColors, 13, "slot-token");
  addText(s, "Slot Tokens  ·  motion-aware part representations", rx + 50, 417, rw - 100, 18, { size: 10, bold: true, align: "center" });
  const assign = box(s, rx + 14, 455, 171, 212, "#F9FBFD", "#A9BCD0", true, "assignment-panel");
  addText(s, "Assignment (Softmax)", rx + 20, 459, 159, 23, { size: 11, bold: true, align: "center" });
  miniHeatmap(s, rx + 37, 493, 7, 7, 18, 16);
  addText(s, "Track", rx + 12, 528, 24, 34, { size: 9, bold: true, align: "center" });
  addText(s, "Slot", rx + 75, 612, 70, 16, { size: 9, bold: true, align: "center" });
  tokenRow(s, rx + 42, 635, 5, 13, queryColors, 10, "assignment-slot-labels");
  const pooled = box(s, rx + 197, 455, 171, 212, "#F9FBFD", "#A9BCD0", true, "pooled-motion-panel");
  addText(s, "Assignment-weighted\nMotion Feature", rx + 204, 460, 157, 39, { size: 11, bold: true, align: "center" });
  for (let i = 0; i < 5; i++) {
    box(s, rx + 218 + i * 25, 527, 18, 63 - i * 4, queryColors[i], "#667788", true, `pooled-feature-${i}`);
  }
  addText(s, "Used for relation reasoning", rx + 210, 618, 145, 28, { size: 9, italic: true, color: C.muted, align: "center" });
  arrow(s, dec, assign, C.stage1l, "elbow", "bottom", "top");
  arrow(s, dec, pooled, C.stage1l, "elbow", "bottom", "top");

  // Stage II: pairwise matrix, relation head, graph, legend.
  box(s, x2 + 10, 54, w2 - 20, 641, "#FFFFFF", C.stage2l, true, "kinematic-reasoning-body");
  addText(s, "Pairwise Slot Features", x2 + 17, 58, w2 - 34, 24, { size: 14, bold: true, align: "center" });
  pairMatrix(s, x2 + 78, 109, 6, 18);
  const rel = labelBox(s, "Relation Head\n(Transformer)", x2 + 37, 247, w2 - 74, 54, "#F1DFF6", C.stage2l, { size: 13, name: "relation-head" });
  const pairAnchor = box(s, x2 + 75, 104, 115, 115, "none", "none", false, "pair-matrix-anchor");
  arrow(s, pairAnchor, rel, C.stage2l);
  addText(s, "Predicted Kinematic Graph + Joint Type", x2 + 20, 320, w2 - 40, 34, { size: 13, bold: true, align: "center" });
  const baseN = node(s, "base", x2 + 93, 369, 78, 34, "#ECECEC", C.base, "graph-base");
  const doorN = node(s, "door", x2 + 32, 476, 78, 36, "#FCE6CF", C.door, "graph-door");
  const drawN = node(s, "drawer", x2 + 156, 476, 78, 36, "#DCE9F7", C.drawer, "graph-drawer");
  arrow(s, baseN, doorN, C.door, "straight", "bottom", "top");
  arrow(s, baseN, drawN, C.drawer, "straight", "bottom", "top");
  addText(s, "R", x2 + 62, 425, 22, 20, { size: 14, bold: true, color: C.door, align: "center" });
  addText(s, "revolute", x2 + 39, 445, 66, 16, { size: 9, color: C.door, align: "center" });
  addText(s, "P", x2 + 184, 425, 22, 20, { size: 14, bold: true, color: C.drawer, align: "center" });
  addText(s, "prismatic", x2 + 168, 445, 58, 16, { size: 9, color: C.drawer, align: "center" });
  box(s, x2 + 28, 557, w2 - 56, 111, "#FCFAFD", "#D9C6E1", true, "part-color-legend");
  const legend = [
    [C.base, "base  (static)"], [C.door, "door  (revolute)"],
    [C.drawer, "drawer  (prismatic)"], [C.slot2, "other part"],
  ];
  legend.forEach(([color, text], i) => {
    box(s, x2 + 43, 570 + i * 22, 14, 14, color, "#6A7180", true);
    addText(s, text, x2 + 65, 567 + i * 22, 144, 20, { size: 10 });
  });

  // Stage III: real scientific visualizations plus editable recovery structure.
  box(s, x3 + 10, 54, w3 - 20, 641, "#FFFFFF", C.stage3l, true, "geometric-recovery-body");
  addText(s, "Relative Motion Estimation", x3 + 18, 57, w3 - 36, 22, { size: 14, bold: true, align: "center" });
  addText(s, "T_rel(t) = T_parent(t)⁻¹ T_child(t)", x3 + 26, 80, w3 - 52, 25, { size: 14, italic: true, align: "center" });
  const gap = 10, innerX = x3 + 19, half = (w3 - 48) / 2;
  box(s, innerX, 112, half, 155, "#FAFCF7", "#B7CE9D", true, "revolute-relative-panel");
  box(s, innerX + half + gap, 112, half, 155, "#FAFCF7", "#B7CE9D", true, "prismatic-relative-panel");
  addText(s, "Revolute (door)", innerX + 4, 115, half - 8, 20, { size: 11, bold: true, align: "center" });
  addText(s, "Prismatic (drawer)", innerX + half + gap + 4, 115, half - 8, 20, { size: 11, bold: true, align: "center" });
  await addImage(s, "relative_motion_revolute.png", innerX + 8, 138, half - 16, 121, "relative_motion_revolute", "contain", { left: 0.13, top: 0.14, right: 0.13, bottom: 0.12 });
  await addImage(s, "relative_motion_prismatic.png", innerX + half + gap + 8, 138, half - 16, 121, "relative_motion_prismatic", "contain", { left: 0.13, top: 0.14, right: 0.13, bottom: 0.12 });
  const opt = labelBox(s, "Geometric Optimization  ·  Replay Error Minimization\nminθ  Σₜ ‖T_rel^obs(t) − T_rel^model(t; θ)‖²_SE(3)", x3 + 26, 278, w3 - 52, 59, "#EAF4E1", C.stage3l, { size: 11, name: "geometric-optimization" });
  const rev = box(s, innerX, 347, half, 135, "#FAFCF7", "#B7CE9D", true, "revolute-joint-panel");
  const pri = box(s, innerX + half + gap, 347, half, 135, "#FAFCF7", "#B7CE9D", true, "prismatic-joint-panel");
  addText(s, "Revolute Joint  ·  Axis + Pivot", innerX + 4, 350, half - 8, 20, { size: 10, bold: true, align: "center" });
  addText(s, "Prismatic Joint  ·  Direction", innerX + half + gap + 4, 350, half - 8, 20, { size: 10, bold: true, align: "center" });
  await addImage(s, "joint_axis_overlay.png", innerX + 12, 374, half - 24, 100, "joint_axis_overlay", "contain", { left: 0.15, top: 0.15, right: 0.15, bottom: 0.12 });
  await addImage(s, "relative_motion_prismatic.png", innerX + half + gap + 12, 374, half - 24, 100, "joint_direction_overlay", "contain", { left: 0.15, top: 0.15, right: 0.15, bottom: 0.12 });
  arrow(s, opt, rev, C.stage3l, "elbow", "bottom", "top");
  arrow(s, opt, pri, C.stage3l, "elbow", "bottom", "top");
  box(s, x3 + 18, 496, w3 - 36, 184, "#F7FBF3", C.stage3l, true, "recovered-model-panel");
  addText(s, "Recovered Articulated Object Model", x3 + 28, 499, w3 - 56, 24, { size: 13, bold: true, align: "center", color: "#32601D" });
  const c1 = 113, c2 = 92, c3 = w3 - 58 - c1 - c2;
  addText(s, "Part Segmentation", x3 + 25, 527, c1, 18, { size: 10, bold: true, align: "center" });
  addText(s, "Kinematic Graph", x3 + 25 + c1, 527, c2, 18, { size: 10, bold: true, align: "center" });
  addText(s, "Joint Parameters", x3 + 25 + c1 + c2, 527, c3, 18, { size: 10, bold: true, align: "center" });
  await addImage(s, "part_segmentation.png", x3 + 29, 547, c1 - 8, 119, "final_model", "contain", { left: 0.13, top: 0.13, right: 0.13, bottom: 0.12 });
  const fb = node(s, "base", x3 + 25 + c1 + 20, 555, 52, 26, "#ECECEC", C.base, "final-base");
  const fd = node(s, "door", x3 + 25 + c1 + 3, 623, 48, 26, "#FCE6CF", C.door, "final-door");
  const fr = node(s, "drawer", x3 + 25 + c1 + 51, 623, 52, 26, "#DCE9F7", C.drawer, "final-drawer");
  arrow(s, fb, fd, C.door, "straight", "bottom", "top");
  arrow(s, fb, fr, C.drawer, "straight", "bottom", "top");
  addText(s, "R", x3 + 25 + c1 + 13, 594, 18, 17, { size: 10, bold: true, color: C.door, align: "center" });
  addText(s, "P", x3 + 25 + c1 + 69, 594, 18, 17, { size: 10, bold: true, color: C.drawer, align: "center" });
  labelBox(s, "door (R)\naxis · pivot\n\ndrawer (P)\ndirection", x3 + 25 + c1 + c2 + 6, 551, c3 - 12, 111, "#FFFFFF", "#C4D5B3", { size: 10, align: "left", name: "joint-parameters" });

  const preview = await p.export({ slide: s, format: "png", scale: 1.5 });
  await fs.writeFile(path.join(OUT, "track2art_method_overview_preview.png"), new Uint8Array(await preview.arrayBuffer()));
  const layout = await s.export({ format: "layout" });
  await fs.writeFile(path.join(OUT, "track2art_method_overview_layout.json"), await layout.text());
  const pptx = await PresentationFile.exportPptx(p);
  await pptx.save(path.join(OUT, "track2art_method_overview.pptx"));
  const soffice = "/Users/lxt/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/soffice";
  try {
    const loProfile = "/tmp/lo-track2art-profile";
    const loOut = "/tmp/lo-track2art-out";
    await fs.mkdir(loProfile, { recursive: true });
    await fs.mkdir(loOut, { recursive: true });
    execFileSync(soffice, [`-env:UserInstallation=file://${loProfile}`, "--headless", "--convert-to", "pdf", "--outdir", loOut, path.join(OUT, "track2art_method_overview.pptx")], { stdio: "inherit" });
    await fs.copyFile(path.join(loOut, "track2art_method_overview.pdf"), path.join(OUT, "track2art_method_overview.pdf"));
  } catch (error) {
    console.warn(`LibreOffice PDF export unavailable: ${error.message}`);
  }
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
