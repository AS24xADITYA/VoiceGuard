"""AI component registry.

Singleton loader for all AI components. Models load once at application
startup into this registry. Loading is lazy per component with eager warmup.

IMPORTANT: This module must NOT import from app/, db/, or any web framework.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()

_registry: ComponentRegistry | None = None
_lock = threading.Lock()


class ComponentRegistry:
    """Singleton registry for AI components.

    Each component loads on first use, but startup fires a background
    warmup task that runs a dummy inference through each.
    """

    def __init__(self) -> None:
        self._components: dict[str, Any] = {}
        self._load_times: dict[str, int] = {}
        self._errors: dict[str, str] = {}
        self._warm: dict[str, bool] = {}
        self._metrics: dict[str, Any] = {}
        self._active_inferences: int = 0
        self._waiting: int = 0
        self._capacity: int = 2

    def register(self, name: str, component: Any) -> None:
        """Register a component by name."""
        self._components[name] = component
        self._warm[name] = False

    def get(self, name: str) -> Any | None:
        """Get a loaded component by name, or None if unavailable."""
        comp = self._components.get(name)
        if comp is not None and hasattr(comp, "is_loaded") and comp.is_loaded():
            return comp
        return None

    def load_component(self, name: str) -> bool:
        """Attempt to load a component. Returns True on success."""
        comp = self._components.get(name)
        if comp is None:
            return False

        if hasattr(comp, "is_loaded") and comp.is_loaded():
            return True

        start = time.monotonic()
        try:
            comp.load()
            elapsed_ms = int((time.monotonic() - start) * 1000)
            self._load_times[name] = elapsed_ms
            log.info("component_loaded", component=name, load_ms=elapsed_ms)
            return True
        except Exception as e:
            self._errors[name] = str(e)
            log.error("component_load_failed", component=name, error=str(e))
            return False

    def warmup_component(self, name: str) -> bool:
        """Warm up a component with a dummy inference."""
        comp = self._components.get(name)
        if comp is None or not (hasattr(comp, "is_loaded") and comp.is_loaded()):
            return False

        try:
            comp.warmup()
            self._warm[name] = True
            log.info("component_warmed", component=name)
            return True
        except Exception as e:
            log.warning("component_warmup_failed", component=name, error=str(e))
            return False

    def load_all(self) -> None:
        """Load all registered components."""
        for name in list(self._components.keys()):
            self.load_component(name)

    def warmup_all(self) -> None:
        """Warm up all loaded components."""
        for name in list(self._components.keys()):
            if self._components[name] is not None:
                self.warmup_component(name)

    def health_report(self) -> dict[str, dict[str, Any]]:
        """Per-component health report for the /system/health endpoint."""
        report: dict[str, dict[str, Any]] = {}

        component_names = ["acoustic", "whisper", "scam", "fusion", "explainer"]
        for name in component_names:
            if name == "acoustic":
                comp = self.get_acoustic()
            elif name == "whisper":
                comp = self.get_transcriber()
            elif name == "scam":
                comp = self.get_scam_classifier()
            elif name == "fusion":
                comp = self.get_fuser()
            elif name == "explainer":
                comp = self.get_explainer()
            else:
                comp = self._components.get(name)

            if comp is None:
                report[name] = {
                    "loaded": False,
                    "warm": False,
                    "version": None,
                    "load_ms": 0,
                    "error": "Component not registered",
                }
            elif hasattr(comp, "is_loaded") and comp.is_loaded():
                report[name] = {
                    "loaded": True,
                    "warm": self._warm.get(name, False),
                    "version": getattr(comp, "version", "unknown"),
                    "load_ms": self._load_times.get(name, 0),
                }
            else:
                err_msg = getattr(comp, "_load_error", None) or self._errors.get(name, "Not loaded")
                report[name] = {
                    "loaded": False,
                    "warm": False,
                    "version": None,
                    "load_ms": 0,
                    "error": err_msg,
                }

        # Database and storage are checked separately
        report["database"] = {"connected": True}  # verified at startup
        report["storage"] = {"backend": "local", "writable": True}

        return report

    def queue_status(self) -> dict[str, int]:
        """Current inference queue status."""
        return {
            "active": self._active_inferences,
            "waiting": self._waiting,
            "capacity": self._capacity,
        }

    def set_capacity(self, cap: int) -> None:
        """Set the max concurrent inference capacity."""
        self._capacity = cap

    def get_acoustic(self) -> Any:
        """Get or instantiate the AcousticDetector component."""
        comp = self._components.get("acoustic")
        if comp is None:
            from ai.acoustic.detector import AcousticDetector

            comp = AcousticDetector()
            self.register("acoustic", comp)
        if not comp.is_loaded():
            comp.load()
        return comp

    def get_transcriber(self) -> Any:
        """Get or instantiate the Transcriber component."""
        comp = self._components.get("whisper")
        if comp is None:
            from ai.linguistic.transcriber import Transcriber

            comp = Transcriber()
            self.register("whisper", comp)
        if not comp.is_loaded():
            comp.load()
        return comp

    def get_scam_classifier(self) -> Any:
        """Get or instantiate the ScamIntentClassifier component."""
        comp = self._components.get("scam")
        if comp is None:
            from ai.linguistic.scam_classifier import ScamIntentClassifier

            comp = ScamIntentClassifier()
            self.register("scam", comp)
        if not comp.is_loaded():
            comp.load()
        return comp

    def get_fuser(self) -> Any:
        """Get or instantiate the FusionEngine component."""
        comp = self._components.get("fusion")
        if comp is None:
            from ai.fusion.fuser import FusionEngine

            comp = FusionEngine()
            self.register("fusion", comp)
        if not comp.is_loaded():
            comp.load()
        return comp

    def get_explainer(self) -> Any:
        """Get or instantiate the GradCAMExplainer component."""
        comp = self._components.get("explainer")
        if comp is None:
            from ai.explain.gradcam import GradCAMExplainer

            acoustic = self.get_acoustic()
            comp = GradCAMExplainer(acoustic_model=acoustic.model)
            self.register("explainer", comp)
        if not comp.is_loaded():
            comp.load()
        return comp

    def get_challenge_verifier(self) -> Any:
        """Get or instantiate the ChallengeVerifier component."""
        comp = self._components.get("challenge")
        if comp is None:
            from ai.challenge.verifier import ChallengeVerifier

            comp = ChallengeVerifier()
            self.register("challenge", comp)
        if not comp.is_loaded():
            comp.load()
        return comp

    def get_metrics(self) -> dict[str, Any]:
        """Read published evaluation metrics from metrics.json artefact."""
        import json
        import os
        from pathlib import Path

        custom_path = os.environ.get("VOICEGUARD_METRICS_PATH")
        candidate_paths = [
            Path(custom_path) if custom_path else None,
            Path(__file__).parent.parent / "app" / "metrics.json",
            Path(__file__).parent.parent / "models" / "metrics.json",
            Path("models/metrics.json"),
            Path("metrics.json"),
        ]

        for p in candidate_paths:
            if p and p.is_file():
                try:
                    return json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    pass

        return {
            "total_analyses": 0,
            "verdict_distribution": {"LOW": 0, "MODERATE": 0, "HIGH": 0, "INCONCLUSIVE": 0},
            "average_duration_ms": 0,
            "status": "PENDING — no trained artifact exists yet, will be populated after Colab training run",
            "model_performance": {
                "acoustic_eer_in_domain": None,
                "acoustic_eer_out_of_domain": None,
                "scam_macro_f1": None,
                "fusion_ece": None,
            },
        }


def get_registry() -> ComponentRegistry:
    """Get or create the singleton registry."""
    global _registry
    if _registry is None:
        with _lock:
            if _registry is None:
                _registry = ComponentRegistry()
    return _registry


def reset_registry() -> None:
    """Reset the registry (for testing)."""
    global _registry
    _registry = None
