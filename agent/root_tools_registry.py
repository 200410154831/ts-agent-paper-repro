import importlib.util
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List

from data.loader import TimeSeriesExamItem


TOOLS_ROOT = Path("/root/tools")


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _default_for_param(name: str):
    defaults = {
        "start": 0,
        "end": None,
        "window": 12,
        "window_size": 12,
        "stride": 1,
        "factor": 2,
        "method": "linear",
        "alpha": 0.3,
        "sigma": 1,
        "lag": 1,
        "lags": [1, 2, 3],
        "q": 0.5,
        "nlags": 20,
        "threshold": 0.05,
        "penalty": 10,
        "n_cp": 2,
        "maxlag": 3,
    }
    return defaults.get(name)


class RootToolsRegistry:
    def __init__(self):
        fproc_module = _load_module("root_tools_fproc", TOOLS_ROOT / "FProc" / "processor.py")
        fnum_module = _load_module("root_tools_fnum", TOOLS_ROOT / "FNum" / "processor.py")
        fdet_module = _load_module("root_tools_fdet", TOOLS_ROOT / "FDet" / "processor.py")
        frel_module = _load_module("root_tools_frel", TOOLS_ROOT / "FRel" / "processor.py")

        self.instances = {
            "FProc": fproc_module.FProc(),
            "FNum": fnum_module.FNum(),
            "FDet": fdet_module.FDet(),
            "FRel": frel_module.FRel(),
        }
        self.registry: Dict[str, Dict[str, Any]] = {}
        self._build_registry()

    def _build_registry(self) -> None:
        for cls_name, inst in self.instances.items():
            for method_name, method in inspect.getmembers(inst, predicate=inspect.ismethod):
                if method_name.startswith("_"):
                    continue
                self.registry[method_name] = {
                    "callable": method,
                    "class": cls_name,
                    "signature": inspect.signature(method),
                }

    @property
    def tool_names(self) -> List[str]:
        return sorted(self.registry.keys())

    @staticmethod
    def _resolve_series_token(value: Any, ts: List[float], ts1: List[float], ts2: List[float]) -> Any:
        if isinstance(value, str):
            key = value.strip().lower()
            if key == "ts":
                return ts if ts else (ts1 if ts1 else ts2)
            if key == "ts1":
                return ts1 if ts1 else (ts if ts else [])
            if key == "ts2":
                return ts2 if ts2 else (ts if ts else [])
        return value

    def execute(self, item: TimeSeriesExamItem, action: str, action_input: Dict[str, Any]) -> Dict[str, Any]:
        if action not in self.registry:
            return {"error": f"unknown_tool:{action}"}
        meta = self.registry[action]
        fn: Callable[..., Any] = meta["callable"]
        sig = meta["signature"]

        ts = item.ts or []
        ts1 = item.ts1 or []
        ts2 = item.ts2 or []

        kwargs: Dict[str, Any] = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue

            if name in {"series", "series1", "series2"} and name in action_input:
                resolved = self._resolve_series_token(action_input[name], ts, ts1, ts2)
                kwargs[name] = resolved
                continue

            if name in action_input:
                kwargs[name] = action_input[name]
                continue

            if name == "series":
                selector = self._resolve_series_token(action_input.get("series", "ts"), ts, ts1, ts2)
                kwargs[name] = selector
                continue
            if name == "series1":
                selector = self._resolve_series_token(action_input.get("series1", "ts1"), ts, ts1, ts2)
                kwargs[name] = selector
                continue
            if name == "series2":
                selector = self._resolve_series_token(action_input.get("series2", "ts2"), ts, ts1, ts2)
                kwargs[name] = selector
                continue

            if param.default is not inspect.Parameter.empty:
                kwargs[name] = param.default
                continue

            fallback = _default_for_param(name)
            if fallback is not None:
                if name == "end" and fallback is None:
                    base = ts if ts else (ts1 if ts1 else ts2)
                    kwargs[name] = len(base)
                else:
                    kwargs[name] = fallback
                continue

            return {"error": f"missing_required_param:{name}"}

        if "end" in kwargs and kwargs["end"] is None:
            base = kwargs.get("series", ts if ts else (ts1 if ts1 else ts2))
            kwargs["end"] = len(base)

        if "series1" in kwargs and not kwargs["series1"]:
            return {"error": "missing_series1"}
        if "series2" in kwargs and not kwargs["series2"]:
            return {"error": "missing_series2"}
        if "series" in kwargs and not kwargs["series"]:
            return {"error": "missing_series"}
        if "series" in kwargs and isinstance(kwargs["series"], str):
            return {"error": f"invalid_series_value:{kwargs['series']}"}
        if "series1" in kwargs and isinstance(kwargs["series1"], str):
            return {"error": f"invalid_series1_value:{kwargs['series1']}"}
        if "series2" in kwargs and isinstance(kwargs["series2"], str):
            return {"error": f"invalid_series2_value:{kwargs['series2']}"}

        try:
            result = fn(**kwargs)
        except Exception as e:
            return {"error": str(e)}
        return {"tool_class": meta["class"], "tool_name": action, "result": result}
