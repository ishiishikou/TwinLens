# Third-party notices

TwinLens downloads model files during the Docker build. They are not committed to this repository.

- OpenCV: Apache-2.0.
- OpenCV Zoo repository: Apache-2.0; model-specific terms apply.
- YuNet directory and model: MIT according to the model directory README.
- SFace directory and model: Apache-2.0 according to the model directory README and LICENSE.
- FastAPI: MIT.
- Uvicorn: BSD-3-Clause.
- NumPy: BSD-3-Clause.
- cryptography: Apache-2.0 OR BSD-3-Clause.

The application intentionally does not use InsightFace-provided pretrained model packs by default. InsightFace code is MIT, while its distributed pretrained models are described by the project as non-commercial research only unless separately licensed.

Before a commercial release, generate a locked dependency SBOM, run a license scanner, pin model artifacts to immutable revisions and hashes, and obtain legal review.
