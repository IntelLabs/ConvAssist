import os
import sys
import json
from pathlib import Path

# CHATGpt Generated Code

class Preferences:
    def __init__(self, app_name: str):
        """Initialize preferences storage."""
        self.app_name = app_name
        self.preferences = {}
        self._ensure_config_dir()
        self.load_all()

    def _ensure_config_dir(self):
        """Ensure the preferences file exists."""
        self.config_dir = self._get_config_dir()
        self.config_file = self.config_dir / f"{self.app_name}_preferences.json"
        if not self.config_file.exists():
            self.save_all()

    def _get_config_dir(self) -> Path:
        """Get the appropriate config directory."""
        if sys.platform == "win32":  # Windows
            return Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / self.app_name
        else:  # macOS/Linux
            return Path.home() / ".config" / self.app_name

    def register_preference(self, key: str, default_value):
        """Register a new preference with a default value if not present."""
        if key not in self.preferences:
            self.preferences[key] = default_value

    def load_all(self):
        """Load all preferences from the config file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    self.preferences = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse {self.config_file}. Using defaults.")

    def load(self, key: str, default=None) -> any:
        """Load a single preference by key."""
        val = self.preferences.get(key, None)
        if val is None:
            self.save(key, default)
            return default
        return val

    def save(self, key: str, value):
        """Save a single preference by key."""
        self.preferences[key] = value
        self._write_to_file()

    def save_all(self):
        """Save all preferences at once."""
        self._write_to_file()

    def _write_to_file(self):
        """Write preferences to the JSON file."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(self.preferences, f, indent=4)

# # Example Usage
# if __name__ == "__main__":
#     prefs = Preferences("MyApp")

#     # Register preferences
#     prefs.register_preference("theme", "light")
#     prefs.register_preference("font_size", 12)

#     # Load and modify preferences
#     print("Current Theme:", prefs.load("theme"))
#     prefs.save("theme", "dark")  # Change theme to dark

#     # Save all preferences
#     prefs.save_all()
