# SBOM


| File                                     | Description                                                                                      |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `[gcbm.cdx.json](gcbm.cdx.json)`         | CycloneDX 1.6 SBOM for packages in `requirements.txt` (versions, licences, SHA-256 wheel hashes) |
| `[licensecheck.json](licensecheck.json)` | licensecheck vs Apache-2.0                                                                       |


Project source: Apache-2.0 (`../LICENSE`, `../NOTICE`).

SBOM generated with **CUDA 11.8** wheels matching Final pins (`torch`/`vision`/`audio` from PyTorch cu118 index; `dgl==2.4.0+cu118` from DGL torch-2.4/cu118 index).

### Generation note


| Item                 | Value                                        |
| -------------------- | -------------------------------------------- |
| Scanner              | `licensecheck` 2025.1.0                      |
| SBOM format          | CycloneDX 1.6 (`gcbm.cdx.json`)              |
| Generated            | 2026-08-08 (CUDA cu118 declared-deps export) |
| Release commit / tag | `v0.1.0` @ `6dbb251`                         |


