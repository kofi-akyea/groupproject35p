"""Model registry for managing trained model versions and metadata."""
import json
import joblib
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

from src.utils.config import MODELS_DIR


class ModelRegistry:
    """Registry for managing model versions and metadata."""
    
    def __init__(self):
        self.registry_path = MODELS_DIR / "registry.json"
        self._registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load the registry from disk or create empty."""
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """Save the registry to disk."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w") as f:
            json.dump(self._registry, f, indent=2)
    
    def register_model(
        self,
        model_name: str,
        model_path: Path,
        metrics: Dict,
        version: str = "1.0.0",
        description: str = ""
    ):
        """Register a trained model with its metadata."""
        self._registry[model_name] = {
            "version": version,
            "path": str(model_path),
            "registered_at": datetime.now().isoformat(),
            "metrics": metrics,
            "description": description,
        }
        self._save_registry()
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get metadata for a registered model."""
        return self._registry.get(model_name)
    
    def list_models(self) -> Dict:
        """List all registered models."""
        return self._registry
    
    def get_latest_model(self, model_name: str) -> Optional[Path]:
        """Get the path to the latest version of a model."""
        info = self.get_model_info(model_name)
        if info:
            return Path(info["path"])
        return None


# Global registry instance
registry = ModelRegistry()
