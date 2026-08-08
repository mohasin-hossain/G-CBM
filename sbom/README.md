# SBOM


| File                                                                       | Description                                                                                          |
| -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `[gcbm.cdx.json](gcbm.cdx.json)`                                           | CycloneDX 1.6 SBOM for all packages in `requirements.txt` (versions, licences, SHA-256 wheel hashes) |
| `[licensecheck.json](licensecheck.json)`                                   | licensecheck vs Apache-2.0                                                                           |
| `[licensecheck_failing_only.txt](licensecheck_failing_only.txt)`           | Incompatible packages only                                                                           |
| `[pytorchcv_upstream_MIT_LICENSE.txt](pytorchcv_upstream_MIT_LICENSE.txt)` | Upstream MIT text for `pytorchcv`                                                                    |


Project source: Apache-2.0 (`../LICENSE`, `../NOTICE`).

SBOM generated with **CUDA 11.8** wheels matching Final pins (`torch`/`vision`/`audio` from PyTorch cu118 index; `dgl==2.4.0+cu118` from DGL torch-2.4/cu118 index).

`pytorchcv` **in** `licensecheck_failing_only.txt`**:** licensecheck leaves the licence field blank (PyPI metadata gap) and marks ✖. Upstream is **MIT** — see `[pytorchcv_upstream_MIT_LICENSE.txt](pytorchcv_upstream_MIT_LICENSE.txt)`; the CycloneDX SBOM and `NOTICE` record MIT. This is not an unresolved Apache conflict.

### Generation note


| Item                 | Value                                        |
| -------------------- | -------------------------------------------- |
| Scanner              | `licensecheck` 2025.1.0                      |
| SBOM format          | CycloneDX 1.6 (`gcbm.cdx.json`)              |
| Generated            | 2026-08-08 (CUDA cu118 declared-deps export) |
| Release commit / tag | Fill in when you cut the public release      |


