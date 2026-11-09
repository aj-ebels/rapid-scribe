"""Distribution names whose dist-info must ship inside the frozen app.

onnx-asr 0.11+ calls importlib.metadata.version("onnx-asr") while importing.
PyInstaller does not collect *.dist-info unless the spec asks for it.
The other names are read at runtime by onnxruntime or huggingface_hub
(feature detection treats a missing dist as "not installed").
"""

METADATA_DISTS = (
    "onnx-asr",
    "onnxruntime",
    "huggingface_hub",
    "numpy",
    "Pillow",
    "pydantic",
    "hf-xet",
    "packaging",
)
