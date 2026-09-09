import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const ROOT = process.cwd();
const OUT = path.join(ROOT, "outputs/track2art_method_overview.pptx");
const PREVIEW = path.join(ROOT, "outputs/track2art_method_overview_preview.png");
const ASSET = path.join(ROOT, "paper_assets/refrigerator045");

const C = {
  ink: "162033", muted: "536273", white: "FFFFFF", line: "B8C4CE",
  s1: "EEF7FD", s1b: "9BCDF2", s1l: "6DAEE0", s1d: "153D74",
  s2: "FBF2FD", s2b: "E7B7F3", s2l: "B879CE", s2d: "572078",
  s3: "F4FAEE", s3b: "CAE8A8", s3l: "90B86C", s3d: "195C25",
  base: "8A8A8A", door: "F28E2B", drawer: "397BC5", extra1: "8CC66A", extra2: "A965D8",
  red: "D9483B", softGray: "F4F6F8", paleBlue: "DCECF8",
};

async function bytes(path) {
  const b = await fs.readFile(path);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}

function box(slide, name, x, y, w, h, fill, stroke, radius = true, width = 1) {
  return slide.shapes.add({
    geometry: radius ? "roundRect" : "rect", name,
    position: { left: x, top: y, width: w, height: h },
    fill, line: { style: "solid", fill: stroke, width },
    borderRadius: radius ? "rounded-md" : undefined,
  });
}

function text(slide, name, value, x, y, w, h, size = 11, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox", name,
    position: { left: x, top: y, width: w, height: h },
    fill: "none", line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = value;
  shape.text.style = {
    fontFamily: "Arial", fontSize: size, bold: Boolean(opts.bold),
    italic: Boolean(opts.italic), color: opts.color || C.ink,
    alignment: opts.align || "center",
  };
  shape.text.verticalAlignment = opts.valign || "middle";
  return shape;
}

function token(slide, name, x, y, color, w = 13, h = 13) {
  return box(slide, name, x, y, w, h, color, color, false, 0.7);
}

function line(slide, name, x, y, w, h, color = C.muted, width = 1.5, arrow = false, dashed = false) {
  const reversed = w < 0 || h < 0;
  const left = w < 0 ? x + w : x;
  const top = h < 0 ? y + h : y;
  return slide.shapes.add({
    geometry: "line", name, position: { left, top, width: Math.abs(w), height: Math.abs(h) }, fill: "none",
    line: { style: dashed ? "dashed" : "solid", fill: color, width,
      ...(arrow ? (reversed
        ? { tail: { type: "triangle", width: "sm", length: "sm" } }
        : { head: { type: "triangle", width: "sm", length: "sm" } }) : {}) },
  });
}

async function image(slide, name, path, x, y, w, h, fit = "contain") {
  const contentType = path.endsWith(".svg") ? "image/svg+xml" : "image/png";
  return slide.images.add({
    blob: await bytes(path), contentType, alt: name, name,
    fit, position: { left: x, top: y, width: w, height: h },
  });
}

function banner(slide, name, title, x, y, w, fill, stroke, color) {
  box(slide, `${name}_bg`, x, y, w, 39, fill, stroke, true, 1.2);
  text(slide, `${name}_title`, title, x + 4, y + 1, w - 8, 35, 20, { bold: true, color });
}

function heatmap(slide, x, y, w, h) {
  const rows = 6, cols = 5, cw = w / cols, ch = h / rows;
  const vals = [
    [0.93, .12, .06, .04, .03], [.82, .20, .08, .04, .02],
    [.08, .88, .12, .05, .02], [.04, .15, .84, .12, .04],
    [.03, .08, .16, .87, .10], [.03, .05, .08, .14, .90],
  ];
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const v = vals[r][c];
    const shade = Math.round(245 - 150 * v).toString(16).padStart(2, "0");
    box(slide, `assignment_heatmap_r${r}_c${c}`, x + c * cw, y + r * ch, cw - 1, ch - 1, `${shade}${shade}F4`, "D7DEEA", false, .4);
  }
  const colors = [C.drawer, C.extra1, C.base, C.door, C.extra2];
  colors.forEach((c, i) => token(slide, `assignment_slot_${i}`, x + i * cw + 3, y + h + 3, c, 8, 8));
  text(slide, "assignment_track_axis", "Track", x - 23, y + 8, 20, h - 8, 8, { bold: true });
  text(slide, "assignment_slot_axis", "Slot", x, y + h + 11, w, 11, 8, { bold: true });
}

async function main() {
  const emphasizedTracks = path.join(ROOT, ".tmp/track2art_method_overview/tracks_4d_emphasis.svg");
  const tracksSvg = (await fs.readFile(path.join(ASSET, "tracks_4d.svg"), "utf8"))
    .replaceAll("#7b8188", "#aeb5bc")
    .replaceAll("#3579b9", "#397bc5")
    .replaceAll("#e8892d", "#f28e2b")
    .replaceAll("stroke-opacity: 0.42; stroke-width: 0.45", "stroke-opacity: 0.82; stroke-width: 0.95");
  await fs.mkdir(path.dirname(emphasizedTracks), { recursive: true });
  await fs.writeFile(emphasizedTracks, tracksSvg);
  const p = Presentation.create({ slideSize: { width: 1280, height: 720 } });
  const slide = p.slides.add();
  slide.background.fill = C.white;

  const m = 7, gap = 5, top = 7, contentY = 50, contentH = 663;
  const w1 = 630, w2 = 269, w3 = 357;
  const x1 = m, x2 = x1 + w1 + gap, x3 = x2 + w2 + gap;

  // Stage surfaces and banners.
  box(slide, "stage1_panel", x1, contentY, w1, contentH, C.s1, C.s1l, true, 1);
  box(slide, "stage2_panel", x2, contentY, w2, contentH, C.s2, C.s2l, true, 1);
  box(slide, "stage3_panel", x3, contentY, w3, contentH, C.s3, C.s3l, true, 1);
  banner(slide, "stage1_banner", "1. Motion-aware Part Discovery", x1, top, w1, C.s1b, C.s1l, C.s1d);
  banner(slide, "stage2_banner", "2. Kinematic Reasoning", x2, top, w2, C.s2b, C.s2l, C.s2d);
  banner(slide, "stage3_banner", "3. Geometric Joint Recovery", x3, top, w3, C.s3b, C.s3l, C.s3d);

  // Main stage transitions, behind content.
  line(slide, "stage1_to_stage2", x1 + w1 - 1, 361, gap + 4, 0, C.s2l, 2.5, true);
  line(slide, "stage2_to_stage3", x2 + w2 - 1, 361, gap + 4, 0, C.s3l, 2.5, true);

  // Stage I left rail.
  const lx = x1 + 10, lw = 230, rx = lx + lw + 12, rw = x1 + w1 - rx - 10;
  box(slide, "rgbd_panel", lx, 60, lw, 155, C.white, "B8D2E6", true, .8);
  text(slide, "rgbd_title", "RGB-D Interaction", lx, 61, lw, 20, 14, { bold: true });
  const rgbPaths = ["rgb_t0.png", "rgb_t1.png", "rgb_t2.png", "rgb_t3.png"];
  const frameW = 49, frameGap = 3, rgbY = 84;
  box(slide, "filmstrip_frame", lx + 9, rgbY - 4, 212, 66, "111111", "111111", false, 1);
  for (let i = 0; i < 12; i++) {
    box(slide, `film_perforation_top_${i}`, lx + 12 + i * 17, rgbY - 1, 10, 5, C.white, C.white, false, 0);
    box(slide, `film_perforation_bottom_${i}`, lx + 12 + i * 17, rgbY + 52, 10, 5, C.white, C.white, false, 0);
  }
  for (let i = 0; i < 4; i++) await image(slide, `rgb_frame_${i}`, `${ASSET}/${rgbPaths[i]}`, lx + 12 + i * (frameW + frameGap), rgbY + 7, frameW, 43, "cover");
  text(slide, "depth_label", "Depth", lx + 8, 147, 64, 14, 9, { bold: true, align: "left" });
  for (let i = 0; i < 4; i++) {
    box(slide, `depth_frame_${i}`, lx + 12 + i * (frameW + frameGap), 164, frameW, 37, ["182B68", "204890", "275FAE", "3579C6"][i], "8DB7E2", false, .6);
    for (let j = 0; j < 5; j++) line(slide, `depth_band_${i}_${j}`, lx + 15 + i * (frameW + frameGap), 168 + j * 6, frameW - 7, 0, ["324691", "3A5FA9", "4B79C2", "6E9DDA", "94BEE8"][j], 1);
  }

  box(slide, "tracker_panel", lx + 3, 225, lw - 6, 52, C.white, C.s1l, true, 1);
  text(slide, "frozen_icon", "❄", lx + 13, 230, 28, 27, 20, { color: C.drawer, bold: true });
  text(slide, "tracker_title", "Persistent Point Tracking", lx + 42, 228, lw - 50, 20, 13.5, { bold: true });
  text(slide, "tracker_subtitle", "Frozen CoTracker3", lx + 42, 248, lw - 50, 15, 10.5, { bold: true });
  line(slide, "rgb_to_tracker", lx + lw / 2, 214, 0, 10, C.s1d, 1.8, true);

  box(slide, "trajectories_panel", lx + 3, 290, lw - 6, 405, C.white, C.s1l, true, 1);
  text(slide, "trajectories_title", "Persistent 4D Point Trajectories", lx + 7, 293, lw - 14, 24, 14.5, { bold: true });
  await image(slide, "trajectories_4d", emphasizedTracks, lx + 8, 320, lw - 16, 365, "cover");
  line(slide, "tracker_to_trajectories", lx + lw / 2, 277, 0, 12, C.s1d, 1.8, true);

  // Stage I architecture rail.
  box(slide, "token_construction_panel", rx, 60, rw, 169, C.white, C.s1l, true, 1);
  text(slide, "token_construction_title", "Track Token Construction", rx, 62, rw, 23, 15.5, { bold: true, color: C.s1d });
  const fboxW = 145;
  box(slide, "appearance_feature_box", rx + 10, 90, fboxW, 66, "F7FAFD", "9DBCE0", true, .8);
  text(slide, "appearance_feature_text", "Tracker Feature\n384D", rx + 16, 94, fboxW - 12, 35, 12.5, { bold: true });
  for (let i = 0; i < 6; i++) token(slide, `appearance_token_${i}`, rx + 21 + i * 19, 133, ["4E82D3", "5F93DC", "75A7E5", "91BBE9", "B2D0EF", "D3E2F5"][i], 14, 13);
  text(slide, "feature_plus", "+", rx + 158, 107, 20, 24, 18, { bold: true });
  box(slide, "geometry_feature_box", rx + 180, 90, fboxW, 66, "F8FBF5", "AFCB91", true, .8);
  text(slide, "geometry_feature_text", "3D Trajectory Geometry\n32D", rx + 186, 94, fboxW - 12, 35, 12.5, { bold: true });
  for (let i = 0; i < 6; i++) token(slide, `geometry_token_${i}`, rx + 191 + i * 19, 133, ["65A94D", "76B55C", "8CC66A", "A8D589", "C3E3A9", "DDEECF"][i], 14, 13);
  text(slide, "per_track_label", "Per-track Tokens", rx + 67, 165, 140, 18, 11.5, { bold: true });
  for (let i = 0; i < 7; i++) token(slide, `per_track_token_${i}`, rx + 89 + i * 23, 190, i % 2 ? "E4EBF2" : "F1F3F5", 16, 15);
  text(slide, "per_track_dim", "416D", rx + 256, 188, 44, 17, 10, { align: "left", bold: true });
  line(slide, "features_to_token_left", rx + 82, 157, 37, 29, C.s1d, 1.2, true);
  line(slide, "features_to_token_right", rx + 250, 157, -48, 29, C.s1d, 1.2, true);

  // Encoder-decoder schematic.
  box(slide, "slot_architecture_panel", rx, 238, rw, 266, "F8FBFE", "9DBCE0", true, 1.2);
  text(slide, "architecture_title", "Encoder + Part-query Decoder", rx + 25, 245, rw - 50, 20, 13.5, { bold: true, color: C.s1d });
  const enc = box(slide, "transformer_encoder", rx + 31, 273, rw - 62, 38, "D8E8F7", "527FB8", true, 1);
  text(slide, "transformer_encoder_text", "Transformer Encoder", rx + 35, 276, rw - 70, 31, 13.5, { bold: true, color: C.s1d });
  const queryColors = [C.drawer, C.extra1, C.base, C.door, C.extra2];
  text(slide, "learnable_queries_label", "Learnable Part Queries", rx + 81, 318, 205, 18, 12, { bold: true });
  queryColors.forEach((c, i) => token(slide, `learnable_query_${i}`, rx + 112 + i * 39, 342, c, 20, 20));
  const dec = box(slide, "transformer_decoder", rx + 31, 379, rw - 62, 39, "D8E8F7", "527FB8", true, 1);
  text(slide, "transformer_decoder_text", "Part-query Transformer Decoder", rx + 35, 382, rw - 70, 32, 13.5, { bold: true, color: C.s1d });
  line(slide, "encoder_to_decoder", rx + 66, 311, 0, 67, C.s1d, 2, true);
  line(slide, "queries_to_decoder", rx + 211, 363, 0, 15, C.s1d, 1.8, true);
  text(slide, "cross_attention_label", "Cross-attn", rx + 2, 340, 82, 18, 9.5, { bold: true });
  line(slide, "cross_attention_arrow", rx + 67, 356, 32, 20, C.s1d, 1.2, true, true);
  queryColors.forEach((c, i) => token(slide, `slot_token_${i}`, rx + 112 + i * 39, 434, c, 20, 20));
  text(slide, "slot_tokens_label", "Motion-aware Part Slots", rx + 63, 460, rw - 126, 21, 12.5, { bold: true });

  // Assignment output and pooled motion features.
  box(slide, "assignment_panel", rx + 6, 514, 184, 173, C.white, "A9C8E5", true, 1);
  text(slide, "assignment_title", "Track-to-slot Assignments", rx + 10, 518, 176, 20, 12, { bold: true });
  heatmap(slide, rx + 43, 545, 132, 105);
  box(slide, "weighted_motion_panel", rx + 199, 514, rw - 205, 173, C.white, "A9C8E5", true, 1);
  text(slide, "weighted_motion_title", "Slot Motion Features", rx + 203, 519, rw - 213, 22, 12, { bold: true });
  [C.base, C.door, C.drawer].forEach((c, i) => {
    token(slide, `weighted_motion_slot_${i}`, rx + 222 + i * 39, 554, c, 23, 25);
    box(slide, `weighted_motion_bar_${i}`, rx + 222 + i * 39, 590, 23, 43 + i * 8, `${c}99`, c, false, .7);
  });
  line(slide, "slots_to_assignment", rx + 140, 482, -40, 31, C.s1d, 2, true);
  line(slide, "assignment_to_weighted", rx + 190, 601, 8, 0, C.s1d, 2, true);

  // Stage II pairwise reasoning.
  const s2x = x2 + 10, s2w = w2 - 20;
  box(slide, "pairwise_features_panel", s2x, 67, s2w, 174, C.white, C.s2l, true, 1);
  text(slide, "pairwise_features_title", "Pairwise Slot Relations", s2x, 70, s2w, 24, 15.5, { bold: true });
  token(slide, "pair_example_i", s2x + 62, 101, C.door, 22, 22);
  text(slide, "pair_example_i_text", "Sᵢ", s2x + 62, 102, 22, 19, 10, { bold: true, color: C.white });
  text(slide, "pair_example_plus", "+", s2x + 89, 101, 20, 22, 16, { bold: true });
  token(slide, "pair_example_j", s2x + 113, 101, C.base, 22, 22);
  text(slide, "pair_example_j_text", "Sⱼ", s2x + 113, 102, 22, 19, 10, { bold: true, color: C.white });
  text(slide, "pair_example_label", "slot pair", s2x + 142, 101, 61, 22, 10.5, { bold: true, color: C.s2d });
  const hmX = s2x + 62, hmY = 132, cell = 23;
  for (let r = 0; r < 5; r++) for (let c = 0; c < 5; c++) {
    const diag = r === c, near = Math.abs(r - c) === 1;
    box(slide, `pairwise_matrix_${r}_${c}`, hmX + c * cell, hmY + r * cell, cell - 1, cell - 1,
      diag ? "D4B1E3" : near ? "E8D5F0" : "F5ECF8", "E2D2E9", false, .35);
  }
  const rel = box(slide, "relation_head", s2x + 30, 258, s2w - 60, 53, "EBD5F1", C.s2l, true, 1.2);
  text(slide, "relation_head_text", "Relation Head", s2x + 34, 263, s2w - 68, 41, 15.5, { bold: true, color: C.s2d });
  line(slide, "pairwise_to_relation", s2x + s2w / 2, 241, 0, 16, C.s2d, 2.2, true);
  box(slide, "edge_existence_output", s2x + 16, 326, 102, 39, "F7ECFA", C.s2l, true, .9);
  text(slide, "edge_existence_text", "Edge existence", s2x + 19, 330, 96, 30, 11.5, { bold: true });
  box(slide, "joint_type_output", s2x + 131, 326, 102, 39, "F7ECFA", C.s2l, true, .9);
  text(slide, "joint_type_text", "Joint type", s2x + 134, 330, 96, 30, 11.5, { bold: true });
  line(slide, "relation_to_edge_output", s2x + 106, 312, -38, 13, C.s2d, 1.8, true);
  line(slide, "relation_to_type_output", s2x + 143, 312, 38, 13, C.s2d, 1.8, true);
  box(slide, "predicted_graph_panel", s2x + 8, 384, s2w - 16, 218, C.white, C.s2l, true, 1);
  text(slide, "predicted_graph_title", "Kinematic Graph & Joint Types", s2x + 15, 389, s2w - 30, 26, 14.5, { bold: true });
  // Graph edges first.
  line(slide, "graph_edge_revolute", s2x + 124, 472, -68, 72, C.door, 3, true);
  line(slide, "graph_edge_prismatic", s2x + 124, 472, 69, 72, C.drawer, 3, true);
  text(slide, "edge_R", "R", s2x + 62, 479, 24, 22, 15, { bold: true, color: C.door });
  text(slide, "edge_P", "P", s2x + 168, 479, 24, 22, 15, { bold: true, color: C.drawer });
  box(slide, "graph_node_base", s2x + 87, 429, 74, 39, "D5D5D5", C.base, true, 1.2);
  text(slide, "graph_node_base_text", "base", s2x + 88, 431, 72, 34, 14, { bold: true, color: "4D4D4D" });
  box(slide, "graph_node_door", s2x + 17, 545, 80, 39, "FCE3CB", C.door, true, 1.2);
  text(slide, "graph_node_door_text", "door", s2x + 18, 547, 78, 34, 14, { bold: true, color: "B55A0A" });
  box(slide, "graph_node_drawer", s2x + 151, 545, 82, 39, "DCE9F7", C.drawer, true, 1.2);
  text(slide, "graph_node_drawer_text", "drawer", s2x + 152, 547, 80, 34, 14, { bold: true, color: "245C9B" });
  line(slide, "outputs_to_graph", s2x + s2w / 2, 366, 0, 17, C.s2d, 2.2, true);

  box(slide, "part_legend", s2x + 20, 618, s2w - 40, 70, "FCFAFD", "CDB8D5", true, .8);
  const legendItems = [[C.base,"base"],[C.door,"door / R"],[C.drawer,"drawer / P"],[C.extra2,"other"]];
  legendItems.forEach((it, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    token(slide, `legend_${i}`, s2x + 34 + col * 108, 630 + row * 27, it[0], 14, 14);
    text(slide, `legend_text_${i}`, it[1], s2x + 53 + col * 108, 626 + row * 27, 82, 22, 10.5, { align: "left", bold: true });
  });

  // Stage III relative motion.
  const s3x = x3 + 9, s3w = w3 - 18;
  box(slide, "relative_motion_panel", s3x, 62, s3w, 280, C.white, C.s3l, true, 1);
  text(slide, "relative_motion_title", "Relative Motion Estimation", s3x, 65, s3w, 24, 15.5, { bold: true });
  text(slide, "relative_motion_formula", "Trel(t) = Tparent(t)⁻¹ Tchild(t)", s3x + 30, 89, s3w - 60, 25, 13, { italic: true });
  const relW = 157;
  box(slide, "relative_revolute_box", s3x + 8, 121, relW, 210, "FBFCFA", "B6CF9D", true, .8);
  text(slide, "relative_revolute_title", "Revolute (door)", s3x + 10, 123, relW - 4, 20, 12.5, { bold: true });
  await image(slide, "relative_motion_revolute", `${ASSET}/relative_motion_revolute.png`, s3x + 13, 147, relW - 10, 174, "contain");
  line(slide, "relative_revolute_axis", s3x + 86, 170, 0, 119, C.door, 3.2);
  box(slide, "relative_revolute_pivot", s3x + 81, 282, 10, 10, C.red, C.red, false, .5);
  box(slide, "relative_prismatic_box", s3x + 174, 121, relW, 210, "FBFCFA", "B6CF9D", true, .8);
  text(slide, "relative_prismatic_title", "Prismatic (drawer)", s3x + 176, 123, relW - 4, 20, 12.5, { bold: true });
  await image(slide, "relative_motion_prismatic", `${ASSET}/relative_motion_prismatic.png`, s3x + 179, 147, relW - 10, 174, "contain");
  line(slide, "relative_prismatic_direction", s3x + 211, 270, 88, -42, C.drawer, 3.2, true);

  box(slide, "geometric_optimization", s3x + 7, 350, s3w - 14, 64, "EAF4E0", C.s3l, true, 1);
  text(slide, "geometric_optimization_title", "Constrained Relative-SE(3) Joint Fitting", s3x + 12, 353, s3w - 24, 22, 13, { bold: true });
  text(slide, "geometric_optimization_formula", "minθ  Σₜ ‖Trelobs(t) − Trelmodel(t; θ)‖²SE(3)", s3x + 17, 377, s3w - 34, 27, 12.5, { italic: true });
  line(slide, "relative_to_optimization", s3x + s3w / 2, 342, 0, 7, C.s3d, 2.2, true);

  box(slide, "joint_revolute_panel", s3x + 8, 422, relW, 112, C.white, "B6CF9D", true, .8);
  text(slide, "joint_revolute_title", "R: axis + pivot", s3x + 10, 424, relW - 4, 20, 12.5, { bold: true, color: C.door });
  await image(slide, "joint_axis_overlay", `${ASSET}/joint_axis_overlay.png`, s3x + 15, 448, relW - 14, 77, "contain");
  box(slide, "joint_prismatic_panel", s3x + 174, 422, relW, 112, C.white, "B6CF9D", true, .8);
  text(slide, "joint_prismatic_title", "P: direction", s3x + 176, 424, relW - 4, 20, 12.5, { bold: true, color: C.drawer });
  await image(slide, "joint_direction_overlay", `${ASSET}/relative_motion_prismatic.png`, s3x + 181, 448, relW - 14, 77, "contain");
  line(slide, "optimization_to_joints", s3x + s3w / 2, 415, 0, 6, C.s3d, 2.2, true);

  box(slide, "recovered_model_panel", s3x, 548, s3w, 154, C.white, C.s3l, true, 1.2);
  text(slide, "recovered_model_title", "Recovered Articulated Model", s3x, 550, s3w, 24, 14.5, { bold: true, color: C.s3d });
  line(slide, "final_divider_1", s3x + 121, 578, 0, 112, "D7E3CC", .8);
  line(slide, "final_divider_2", s3x + 230, 578, 0, 112, "D7E3CC", .8);
  text(slide, "final_seg_title", "Part Segmentation", s3x + 4, 576, 113, 20, 10.5, { bold: true });
  await image(slide, "final_model", `${ASSET}/part_segmentation.png`, s3x + 6, 595, 112, 94, "contain");
  text(slide, "final_graph_title", "Kinematic Graph", s3x + 125, 576, 101, 20, 10.5, { bold: true });
  line(slide, "final_edge_r", s3x + 174, 625, -27, 31, C.door, 1.8, true);
  line(slide, "final_edge_p", s3x + 174, 625, 28, 31, C.drawer, 1.8, true);
  box(slide, "final_base", s3x + 154, 602, 42, 22, "D5D5D5", C.base, true, .7);
  text(slide, "final_base_text", "base", s3x + 155, 603, 40, 19, 8, { bold: true });
  box(slide, "final_door", s3x + 128, 657, 46, 22, "FCE3CB", C.door, true, .7);
  text(slide, "final_door_text", "door", s3x + 129, 658, 44, 19, 8, { bold: true, color: "B55A0A" });
  box(slide, "final_drawer", s3x + 186, 657, 50, 22, "DCE9F7", C.drawer, true, .7);
  text(slide, "final_drawer_text", "drawer", s3x + 187, 658, 48, 19, 8, { bold: true, color: "245C9B" });
  text(slide, "final_params_title", "Joint Geometry", s3x + 234, 576, 101, 20, 10.5, { bold: true });
  text(slide, "final_params_r", "R: axis + pivot", s3x + 241, 608, 90, 25, 11.5, { align: "left", bold: true, color: C.door });
  text(slide, "final_params_p", "P: direction", s3x + 241, 644, 90, 25, 11.5, { align: "left", bold: true, color: C.drawer });
  line(slide, "joints_to_final", s3x + s3w / 2, 535, 0, 12, C.s3d, 2.2, true);

  // Source note: all embedded scientific images are repository-generated assets.
  const notes = slide.addNotes?.bind(slide);
  if (notes) notes("[Sources]\n- Repository-generated assets: paper_assets/refrigerator045/*.png\n[/Sources]");

  await fs.mkdir(path.join(ROOT, "outputs"), { recursive: true });
  const png = await p.export({ slide, format: "png", scale: 1.5 });
  await fs.writeFile(PREVIEW, new Uint8Array(await png.arrayBuffer()));
  const layout = await slide.export({ format: "layout" });
  await fs.mkdir(path.join(ROOT, ".tmp/track2art_method_overview"), { recursive: true });
  await fs.writeFile(path.join(ROOT, ".tmp/track2art_method_overview/slide.layout.json"), await layout.text());
  const pptx = await PresentationFile.exportPptx(p);
  await pptx.save(OUT);
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
