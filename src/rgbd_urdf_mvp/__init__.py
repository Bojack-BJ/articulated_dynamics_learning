"""RGB-D to URDF MVP scaffold."""

from .pipeline import RGBDToURDFPipeline
from .core.serialization import load_episode

__all__ = ["RGBDToURDFPipeline", "load_episode"]
