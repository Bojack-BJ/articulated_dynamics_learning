from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET


def write_xml_tree(tree: ET.ElementTree, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass
    tree.write(path, encoding="utf-8", xml_declaration=True)
