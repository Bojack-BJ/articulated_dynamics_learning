from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import textwrap
import unittest
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image


class PartMaskAnnotationUiTest(unittest.TestCase):
    def test_mask_status_exposes_mask_file_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask_path = root / "assets" / "masks" / "manual_partmask" / "view_0" / "frame_0000_mask.png"
            mask_path.parent.mkdir(parents=True)
            Image.fromarray(np.asarray([[0, 1], [1, 0]], dtype=np.uint16), mode="I;16").save(mask_path)
            episode_path = root / "episode.json"
            episode_path.write_text(
                textwrap.dedent(
                    """
                    {
                      "object_instance_id": "box",
                      "category": "box",
                      "frames": [
                        {
                          "timestamp_s": 0.0,
                          "rgb_path": "assets/view_0/frame_0000_rgb.png",
                          "part_mask_path": "assets/masks/manual_partmask/view_0/frame_0000_mask.png",
                          "part_mask_paths_by_view": ["assets/masks/manual_partmask/view_0/frame_0000_mask.png"]
                        }
                      ]
                    }
                    """
                ).strip(),
                encoding="utf-8",
            )

            module = _load_annotation_server_module()
            state = module.AnnotationState(
                episode_path=episode_path,
                output_dir=None,
                output_episode=None,
                default_view_index=0,
                default_frame_index=0,
                propagate_python=None,
                sam2_root=Path("."),
                sam2_config="configs/sam2.1/sam2.1_hiera_t.yaml",
                sam2_checkpoint=None,
                sam2_device="cpu",
                cotracker_repo=None,
                cotracker_checkpoint=None,
            )
            frame_status = state.mask_status(0)["frames"][0]
            self.assertTrue(frame_status["exists"])
            self.assertEqual(frame_status["mask_size_bytes"], mask_path.stat().st_size)
            self.assertEqual(frame_status["mask_mtime_ns"], mask_path.stat().st_mtime_ns)

    def test_save_mask_strips_unknown_part_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episode_path = root / "episode.json"
            episode_path.write_text(
                textwrap.dedent(
                    """
                    {
                      "object_instance_id": "box",
                      "category": "box",
                      "frames": [
                        {
                          "timestamp_s": 0.0,
                          "rgb_path": "assets/view_0/frame_0000_rgb.png"
                        }
                      ]
                    }
                    """
                ).strip(),
                encoding="utf-8",
            )

            module = _load_annotation_server_module()
            state = module.AnnotationState(
                episode_path=episode_path,
                output_dir=None,
                output_episode=None,
                default_view_index=0,
                default_frame_index=0,
                propagate_python=None,
                sam2_root=Path("."),
                sam2_config="configs/sam2.1/sam2.1_hiera_t.yaml",
                sam2_checkpoint=None,
                sam2_device="cpu",
                cotracker_repo=None,
                cotracker_checkpoint=None,
            )
            labels = np.asarray([[1, 2], [3, 0]], dtype=np.uint16)
            result = state.save_mask(
                {
                    "frame_index": 0,
                    "view_index": 0,
                    "width": 2,
                    "height": 2,
                    "mask_data_url": module._encode_label_png(labels),
                    "parts": [
                        {"part_id": 1, "name": "base", "role": "base", "color": "#62d26f"},
                        {"part_id": 2, "name": "moving", "role": "moving", "color": "#a56de2"},
                    ],
                    "prompts": [],
                }
            )
            saved = np.asarray(Image.open(result["mask_path"]), dtype=np.uint16)
            self.assertEqual(saved.tolist(), [[1, 2], [0, 0]])
            self.assertEqual(result["part_pixels"], {"1": 1, "2": 1})

    def test_embedded_javascript_history_smoke(self) -> None:
        node = shutil.which("node")
        if node is None:
            self.skipTest("node is not installed")

        source = Path("scripts/serve_partmask_annotation.py").read_text()
        match = re.search(r"<script>\n(.*?)\n</script>", source, re.S)
        self.assertIsNotNone(match)
        script = match.group(1).replace(
            "\ninit().catch(err => setStatus(`Failed to load: ${err.message}`));",
            "",
        )
        harness = f"""
        const vm = require('vm');
        const elements = new Map();
        function makeCtx() {{
          return {{
            createImageData: (w, h) => ({{ data: new Uint8ClampedArray(w * h * 4) }}),
            putImageData: () => {{}},
            clearRect: () => {{}},
            drawImage: () => {{}},
            getImageData: (x, y, w, h) => ({{ data: new Uint8ClampedArray(w * h * 4) }}),
            beginPath: () => {{}}, arc: () => {{}}, stroke: () => {{}}, moveTo: () => {{}}, lineTo: () => {{}}, strokeRect: () => {{}}, setLineDash: () => {{}},
            lineWidth: 1, strokeStyle: '', fillStyle: '',
          }};
        }}
        function element(id='') {{
          if (!elements.has(id)) {{
            const el = {{
              id, value: '', textContent: '', innerHTML: '', checked: false, style: {{}}, className: '', dataset: {{}}, children: [], onclick: null, oninput: null, onchange: null,
              appendChild(child) {{ this.children.push(child); return child; }},
              querySelector() {{ return element(`${{id}}:button`); }},
              classList: {{ toggle: () => {{}} }},
              getContext: () => makeCtx(),
              getBoundingClientRect: () => ({{ left: 0, top: 0, width: 4, height: 4 }}),
              addEventListener(type, fn) {{ this[`on${{type}}`] = fn; }},
              toDataURL: () => 'data:image/png;base64,',
            }};
            elements.set(id, el);
          }}
          return elements.get(id);
        }}
        const sandbox = {{
          console, elements,
          Uint16Array, Uint8Array, Int32Array, Uint8ClampedArray, Math, Number, Boolean, JSON, Promise, Date, setInterval: () => 0,
          document: {{
            getElementById: element,
            querySelectorAll: () => [],
            createElement: tag => element(`created:${{tag}}:${{Math.random()}}`),
          }},
          window: {{ addEventListener(type, fn) {{ this[`on${{type}}`] = fn; }} }},
          fetch: async () => ({{ json: async () => ({{}}), headers: {{ get: () => 'application/json' }} }}),
          Image: function() {{}},
          createImageBitmap: async () => ({{}}),
        }};
        const code = {script!r} + `
        function assert(cond, msg) {{ if (!cond) throw new Error(msg); }}
        state = {{frame_count: 10}};
        parts = [{{part_id:1,name:'base',role:'base',color:'#62d26f'}}];
        currentPart = 1;
        width = 4; height = 4;
        labels = new Uint16Array(width * height);
        prompts = [];
        history = [];
        renderParts = () => {{}};
        drawOverlay = () => {{}};
        drawPrompts = () => {{}};
        renderPromptList = () => {{}};
        setStatus = (text) => {{ lastStatus = text; }};
        labels[5] = 1;
        pushHistory();
        labels[5] = 2;
        restoreHistory(history.pop());
        assert(labels[5] === 1, 'undo should restore labels');
        pushHistory();
        prompts.push({{type:'bbox', part_id:1, x1:0, y1:0, x2:2, y2:2}});
        restoreHistory(history.pop());
        assert(prompts.length === 0, 'undo should restore prompts');
        document.getElementById('textPrompt').value = 'lid';
        document.getElementById('addTextPromptBtn').onclick();
        assert(prompts.length === 1 && history.length === 1, 'text prompt should push history');
        restoreHistory(history.pop());
        assert(prompts.length === 0, 'undo should remove text prompt');
        document.getElementById('addPartBtn').onclick();
        assert(parts.length === 2 && currentPart === 2, 'add part should work');
        restoreHistory(history.pop());
        assert(parts.length === 1 && currentPart === 1, 'undo should restore parts/current part');
        labels.fill(0);
        labels[0] = 1; labels[1] = 1; labels[15] = 1;
        keepLargestCurrentPart();
        assert(labels[0] === 1 && labels[1] === 1 && labels[15] === 0, 'keep largest should remove speckle');
        restoreHistory(history.pop());
        assert(labels[15] === 1, 'undo should restore keep-largest change');
        assert(
          maskEntryKey({{exists:true, mask_path:'/tmp/mask.png', mask_mtime_ns:12, mask_size_bytes:34}}) === '/tmp/mask.png:12:34',
          'mask entry key should include path, mtime, and size'
        );
        assert(maskEntryKey({{exists:false, mask_path:'/tmp/mask.png'}}) === null, 'missing masks should not produce keys');
        `;
        vm.runInNewContext(code, sandbox, {{ filename: 'partmask_ui_history_smoke.js' }});
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "partmask_ui_history_smoke.js"
            path.write_text(textwrap.dedent(harness))
            result = subprocess.run([node, str(path)], text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def _load_annotation_server_module():
    path = Path("scripts/serve_partmask_annotation.py").resolve()
    spec = importlib.util.spec_from_file_location("serve_partmask_annotation_for_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
