import os
import sys
import tempfile

# Make `import app` work with plain `pytest`, not only `python -m pytest`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Runs before any test module imports `app`, so config picks up a throwaway database
# instead of leaving test.db behind in the project folder.
os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(prefix="mafl-"), "test.db")
