import pathlib

PROJECT_DIR = pathlib.Path(__file__).parent.parent.parent.resolve()

EXAMPES_DIR = PROJECT_DIR / "examples"

# The path to the exporter module directory.
EXPORTER_ROOT_DIR = PROJECT_DIR / "exporter"

EXPORTER_TESTS_DIR = EXPORTER_ROOT_DIR / "tests"
