import sys
from unittest.mock import MagicMock

# Stub osxphotos on non-Mac platforms so tests run during development on Windows
if sys.platform != "darwin":
    sys.modules["osxphotos"] = MagicMock()
