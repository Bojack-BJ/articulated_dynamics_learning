from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco

from ..core.serialization import save_json


@dataclass(slots=True)
class ARXX5PrepareConfig:
    output_dir: str | Path
    source_dir: str | Path | None = None
    repo_url: str = "https://github.com/ARXroboticsX/ARX_Model.git"
    validate_mujoco: bool = True


class ARXX5ModelPreparer:
    def prepare(self, config: ARXX5PrepareConfig) -> Path:
        output_dir = Path(config.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        source_x5a = self._resolve_source(config, output_dir)
        package_dir = output_dir / "X5A"
        if package_dir.exists():
            shutil.rmtree(package_dir)
        shutil.copytree(source_x5a, package_dir)

        urdf_path = package_dir / "urdf" / "X5A.urdf"
        text = urdf_path.read_text(encoding="utf-8")
        patched = text.replace("package://X5A/meshes/", "../meshes/")
        patched_path = package_dir / "urdf" / "X5A.mujoco.urdf"
        patched_path.write_text(patched, encoding="utf-8")

        validation: dict[str, Any] = {"requested": bool(config.validate_mujoco)}
        if config.validate_mujoco:
            model = mujoco.MjModel.from_xml_path(str(patched_path))
            joints = [
                {
                    "id": int(jid),
                    "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"joint_{jid}",
                    "type": int(model.jnt_type[jid]),
                    "range": [float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])],
                }
                for jid in range(model.njnt)
            ]
            validation.update({"loaded": True, "nq": int(model.nq), "nv": int(model.nv), "joints": joints})

        manifest_path = output_dir / "arx_x5_model_manifest.json"
        save_json(
            {
                "source": "arx-x5-model-preparer",
                "source_dir": str(source_x5a),
                "package_dir": str(package_dir),
                "urdf_path": str(patched_path),
                "validation": validation,
            },
            manifest_path,
        )
        return manifest_path

    def _resolve_source(self, config: ARXX5PrepareConfig, output_dir: Path) -> Path:
        if config.source_dir is not None:
            source = Path(config.source_dir).expanduser().resolve()
        else:
            clone_dir = output_dir / "_ARX_Model_source"
            if clone_dir.exists():
                shutil.rmtree(clone_dir)
            subprocess.run(
                ["git", "clone", "--depth", "1", str(config.repo_url), str(clone_dir)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            source = clone_dir
        x5a = source / "X5" / "X5A"
        if not (x5a / "urdf" / "X5A.urdf").is_file():
            raise FileNotFoundError(f"Could not find X5/X5A/urdf/X5A.urdf under {source}")
        return x5a

