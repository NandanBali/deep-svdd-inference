"""Download the explicitly selected official release and verify its BagIt checksums."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

UUID = "abd2f5d8-e268-4ff0-84fb-f2f7b3ca3e8f"
API = f"https://api.rdm.uq.edu.au/datasets/{UUID}"
NAME = "NF-UNSW-NB15-v3"
EXPECTED_SHA1 = "2cba4879ead17c51e3268a01c06af41186cd90e4"


def digest(path: Path, algorithm: str = "sha256") -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def acquire(root: Path) -> Path:
    data, sources = root / "data", root / "sources"
    data.mkdir(parents=True, exist_ok=True)
    sources.mkdir(parents=True, exist_ok=True)
    archive, csv = data / f"{NAME}.zip", data / f"{NAME}.csv"
    if not archive.exists():
        with urllib.request.urlopen(f"{API}/file-details", timeout=60) as r:
            metadata = json.load(r)
        if metadata["name"] != NAME or metadata["accessType"] != "OPEN":
            raise ValueError("Official dataset identity/access changed; inspect before proceeding")
        (sources / "uq_file-details.json").write_text(json.dumps(metadata, indent=2) + "\n")
        with urllib.request.urlopen(f"{API}/download", timeout=60) as r:
            info = json.load(r)
        if not info.get("canDownload"):
            raise ValueError("Official dataset download is unavailable")
        temporary = archive.with_suffix(".zip.part")
        with urllib.request.urlopen(info["url"], timeout=120) as r, temporary.open("wb") as f:
            shutil.copyfileobj(r, f, 4 * 1024 * 1024)
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            base = Path(name).name
            if base in {f"{NAME}.csv", "NetFlow_v3_Features.csv", "manifest-sha1.txt", "bag-info.txt"}:
                dest = (data if base == f"{NAME}.csv" else sources) / base
                if not dest.exists():
                    with z.open(name) as src, dest.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
    if digest(csv, "sha1") != EXPECTED_SHA1:
        raise ValueError("Dataset differs from the pinned official BagIt payload")
    for line in (sources / "manifest-sha1.txt").read_text().splitlines():
        expected, name = line.split()
        base = Path(name).name
        dest = (data if base == csv.name else sources) / base
        if digest(dest, "sha1") != expected:
            raise ValueError(f"BagIt checksum mismatch: {base}")
    provenance = {
        "release": NAME, "doi": "10.48610/6E0EDA1", "uuid": UUID,
        "landing_page": f"https://rdm.uq.edu.au/files/{UUID}",
        "download_endpoint": f"{API}/download", "archive_sha256": digest(archive),
        "csv_sha256": digest(csv), "csv_sha1": EXPECTED_SHA1,
        "csv_bytes": csv.stat().st_size, "bagit_checksums_verified": True,
        "choice": "No project dataset existed. Selected official v3 for timestamp-supported separation, then verified actual schema.",
        "license": "Permitted Re-Use with Commercial Use Restriction; see official metadata",
    }
    (sources / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return csv
