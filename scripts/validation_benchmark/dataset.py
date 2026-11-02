"""Frozen MassBank source acquisition and deterministic, label-only selection."""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from matchms import Spectrum

SOURCE_COMMIT = "705afb7bccc3b2c42410a744eef73674716a60ef"
SOURCE_URL = (
    f"https://codeload.github.com/MassBank/MassBank-data/tar.gz/{SOURCE_COMMIT}"
)
SOURCE_SHA256 = "cb30a4aab9d06dc5978c08e73397106a104955f1c31a57671ee5c15025985529"


@dataclass
class Record:
    """One experimental record; chemical identity is independent of accession."""

    accession: str
    inchikey: str
    precursor_mz: float
    mz: np.ndarray
    intensities: np.ndarray
    metadata: dict[str, Any]
    source_path: str
    source_sha256: str

    @property
    def connectivity(self) -> str:
        """Return the declared connectivity block without asserting stereo selectivity."""
        return self.inchikey.split("-")[0]

    def spectrum(self, identifier: str, blind: bool = False) -> Spectrum:
        """Create a float64 spectrum, withholding query truth from the pipeline.

        Parameters
        ----------
        identifier : str
            Stable, unique experimental-record or blinded query ID.
        blind : bool
            Remove answer-bearing chemical fields for a query.

        Returns
        -------
        Spectrum
            New independent spectrum; source arrays are never mutated.

        Examples
        --------
        >>> # record.spectrum('query_0001', blind=True)
        """
        metadata = {
            "id": identifier,
            "compound_name": identifier
            if blind
            else self.metadata.get("name", identifier),
            "precursor_mz": self.precursor_mz,
            "adduct": self.metadata.get("adduct"),
            "charge": 1,
            "ionmode": "positive",
            "instrument": self.metadata.get("instrument"),
            "collision_energy": self.metadata.get("collision_energy"),
            "source_accession": None if blind else self.accession,
        }
        if not blind:
            metadata.update(
                {
                    key: self.metadata[key]
                    for key in ("formula", "smiles", "inchi")
                    if self.metadata.get(key)
                }
            )
            metadata["inchikey"] = self.inchikey
        metadata = {key: value for key, value in metadata.items() if value is not None}
        return Spectrum(
            self.mz.copy(),
            self.intensities.copy(),
            metadata=metadata,
            metadata_harmonization=False,
        )


def _parse(content: bytes, source_path: str) -> Record:
    """Parse original MassBank metadata and PK$PEAK rows without chemistry inference."""
    text = content.decode("utf8")
    fields: dict[str, list[str]] = defaultdict(list)
    peak_lines = []
    in_peaks = False
    # Source lines are a text interchange format, not a scoring-array hot loop.
    for line in text.splitlines():
        if line.startswith("PK$PEAK:"):
            in_peaks = True
        elif in_peaks:
            if line == "//":
                break
            peak_lines.append(line)
        elif ": " in line:
            key, value = line.split(": ", 1)
            fields[key].append(value)

    def first(tag: str, default: str = "") -> str:
        return fields.get(tag, [default])[0]

    def sub(tag: str, name: str) -> str:
        return next(
            (
                v[len(name) + 1 :]
                for v in fields.get(tag, [])
                if v.startswith(name + " ")
            ),
            "",
        )

    raw_peaks = np.fromstring(" ".join(peak_lines), sep=" ", dtype=np.float64)
    if raw_peaks.size % 3:
        raise ValueError(f"Malformed peak triples: {source_path}")
    peaks = raw_peaks.reshape((-1, 3))
    mass_text = sub("MS$FOCUSED_ION", "PRECURSOR_M/Z")
    metadata = {
        "license": first("LICENSE"),
        "authors": first("AUTHORS"),
        "copyright": first("COPYRIGHT"),
        "name": first("CH$NAME"),
        "formula": first("CH$FORMULA"),
        "smiles": first("CH$SMILES"),
        "inchi": first("CH$IUPAC"),
        "instrument": first("AC$INSTRUMENT"),
        "instrument_type": first("AC$INSTRUMENT_TYPE"),
        "ionization": sub("AC$MASS_SPECTROMETRY", "IONIZATION"),
        "polarity": sub("AC$MASS_SPECTROMETRY", "ION_MODE"),
        "ms_level": sub("AC$MASS_SPECTROMETRY", "MS_TYPE"),
        "adduct": sub("MS$FOCUSED_ION", "PRECURSOR_TYPE"),
        "collision_energy": sub("AC$MASS_SPECTROMETRY", "COLLISION_ENERGY"),
        "confidence": sub("COMMENT", "CONFIDENCE"),
        "deprofile": sub("MS$DATA_PROCESSING", "DEPROFILE"),
        "processing": fields.get("MS$DATA_PROCESSING", []),
        "deprecated": bool(fields.get("DEPRECATED")),
        "laboratory": source_path.split("/")[0],
    }
    return Record(
        first("ACCESSION"),
        sub("CH$LINK", "INCHIKEY"),
        float(mass_text) if mass_text else float("nan"),
        peaks[:, 0].copy(),
        peaks[:, 1].copy(),
        metadata,
        source_path,
        hashlib.sha256(content).hexdigest(),
    )


def read_record(path: Path) -> Record:
    """Read one original redistributable MassBank record.

    Parameters
    ----------
    path : Path
        Original text record.

    Returns
    -------
    Record
        Peaks and unmodified declared chemical/source metadata.

    Examples
    --------
    >>> # read_record(Path('MSBNK-Eawag-EA262601.txt'))
    """
    lab = path.stem.split("-")[1]
    return _parse(path.read_bytes(), f"{lab}/{path.name}")


def peak_hash(record: Record) -> str:
    """Hash exact m/z plus base-peak-normalized float64 intensities rounded at 12 decimals.

    Parameters
    ----------
    record : Record
        Experimental peak list.

    Returns
    -------
    str
        Scale-invariant duplicate-screen hash, not an identity proof.

    Examples
    --------
    >>> # peak_hash(record)
    """
    maximum = float(np.max(record.intensities)) if record.intensities.size else 0.0
    intensities = (
        np.round(record.intensities / maximum, 12)
        if maximum > 0
        else record.intensities
    )
    return hashlib.sha256(
        np.asarray(record.mz, dtype="<f8").tobytes()
        + np.asarray(intensities, dtype="<f8").tobytes()
    ).hexdigest()


def manifest_digest(manifest: dict[str, Any]) -> str:
    """Identify canonical manifest bytes including schema and selection settings.

    Parameters
    ----------
    manifest : dict
        Dataset manifest.

    Returns
    -------
    str
        SHA-256 of canonical JSON.

    Examples
    --------
    >>> len(manifest_digest({}))
    64
    """
    return hashlib.sha256(
        json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def verify_manifest(manifest: dict[str, Any], expected: str) -> None:
    """Reject unsupported schema versions or modified frozen selection inputs.

    Parameters
    ----------
    manifest : dict
        Frozen manifest data.
    expected : str
        Independently saved expected digest.

    Returns
    -------
    None
        Raises ValueError on mismatch.

    Examples
    --------
    >>> verify_manifest({'schema_version':1}, manifest_digest({'schema_version':1}))
    """
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported dataset schema version")
    if manifest_digest(manifest) != expected:
        raise ValueError("Frozen manifest digest mismatch")


def _eligibility(record: Record) -> list[str]:
    """Apply prespecified source/measurement/label rules before any scores exist."""
    m = record.metadata
    checks = {
        "license_not_CC_BY_or_CC0": m["license"] not in {"CC BY", "CC0"},
        "not_standard_compound": m["confidence"].lower() != "standard compound",
        "not_positive_ESI_MS2": m["ms_level"] != "MS2"
        or m["polarity"] != "POSITIVE"
        or ("ESI" not in m["instrument_type"] and m["ionization"] != "ESI"),
        "not_protonated": m["adduct"] != "[M+H]+",
        "invalid_full_inchikey": not bool(
            re.fullmatch("[A-Z]{14}-[A-Z]{10}-[A-Z]", record.inchikey)
        ),
        "missing_collision_energy": not m["collision_energy"],
        "invalid_precursor": not np.isfinite(record.precursor_mz)
        or record.precursor_mz <= 0,
        "invalid_peaks": record.mz.size == 0
        or not np.all(np.isfinite(record.mz))
        or not np.all(np.isfinite(record.intensities))
        or np.any(np.diff(record.mz) <= 0)
        or np.any(record.intensities < 0),
        "deprecated": m["deprecated"],
    }
    return [key for key, failed in checks.items() if failed]


def _load_archive(path: Path) -> tuple[list[Record], dict[str, Any]]:
    """Inventory both prespecified laboratory cohorts; count all other source records."""
    records = []
    counts: Counter[str] = Counter()
    exclusions = []
    with tarfile.open(path) as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".txt"):
                continue
            counts["archive_text_records"] += 1
            relative = member.name.split("/", 1)[1]
            if relative.split("/")[0] not in {"Athens_Univ", "Eawag"}:
                counts["outside_prespecified_sources"] += 1
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"Unreadable archive member {relative}")
            try:
                record = _parse(handle.read(), relative)
                reasons = _eligibility(record)
            except (ValueError, UnicodeError) as error:
                exclusions.append(
                    {
                        "source_path": relative,
                        "reasons": ["parse_error"],
                        "detail": str(error),
                    }
                )
                continue
            if reasons:
                exclusions.append(
                    {
                        "accession": record.accession,
                        "source_path": relative,
                        "reasons": reasons,
                    }
                )
            else:
                records.append(record)
    counts["eligible_source_records"] = len(records)
    counts["excluded_within_source_records"] = len(exclusions)
    return records, {"counts": dict(counts), "exclusions": exclusions}


def _fetch(cache: Path) -> Path:
    """Download and verify the pinned archive outside Git, without trusting cached bytes."""
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"massbank-{SOURCE_COMMIT}.tar.gz"
    if not path.exists():
        temporary = path.with_suffix(".download")
        with (
            urllib.request.urlopen(SOURCE_URL, timeout=60) as source,
            temporary.open("wb") as destination,
        ):
            while chunk := source.read(1024 * 1024):
                destination.write(chunk)
        temporary.replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError(
            "Source archive checksum mismatch; preserve bytes and investigate"
        )
    return path


def _select(
    records: list[Record], development: int, test: int
) -> tuple[list[Record], list[Record], dict[str, Any]]:
    """Select distinct connectivity groups by hash and retain the complete eligible order."""
    by_key: dict[str, dict[str, list[Record]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_key[record.inchikey][record.metadata["laboratory"]].append(record)
    truths: dict[str, list[str]] = defaultdict(list)
    for key, labs in by_key.items():
        if "Athens_Univ" in labs and "Eawag" in labs:
            truths[key[:14]].append(key)

    def key_order(value: str) -> str:
        """Order eligible groups independently of all matching outcomes."""
        return hashlib.sha256(("massflow-mvb1-v1|" + value).encode()).hexdigest()

    ordered = sorted(truths, key=key_order)
    if len(ordered) < development + test:
        raise ValueError(
            f"Only {len(ordered)} qualifying groups, need {development + test}"
        )
    chosen = ordered[development : development + test]
    queries = []
    for group in chosen:
        full = min(
            truths[group],
            key=lambda value: hashlib.sha256(
                ("massflow-mvb1-truth-v1|" + value).encode()
            ).hexdigest(),
        )
        queries.append(min(by_key[full]["Athens_Univ"], key=lambda r: r.accession))
    library = [r for r in records if r.metadata["laboratory"] == "Eawag"]
    return (
        queries,
        library,
        {
            "eligible_connectivity_groups": len(ordered),
            "ordered_groups": ordered,
            "development_groups": ordered[:development],
            "test_groups": chosen,
            "absent_groups": chosen[: test // 2],
        },
    )


def audit_leakage(queries: list[Record], library: list[Record]) -> dict[str, Any]:
    """Screen duplicate spectra/identities and cross-source near matches without tuning.

    Parameters
    ----------
    queries : list[Record]
        Fixed query panel selected before scoring.
    library : list[Record]
        Frozen source library.

    Returns
    -------
    dict
        Exact hashes, repeated chemical identities and potential near duplicates.

    Examples
    --------
    >>> audit_leakage([],[])['exact_cross_pairs']
    []
    """
    from matchms.similarity import CosineGreedy

    identifiers = [r.accession for r in library]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Ambiguous repeated source accession")
    hashes: dict[str, list[str]] = defaultdict(list)
    identities = Counter(r.inchikey for r in library)
    for record in library:
        hashes[peak_hash(record)].append(record.accession)
    exact: list[dict[str, Any]] = []
    near = []
    scorer = CosineGreedy(tolerance=0.005)
    masses = np.asarray([r.precursor_mz for r in library], dtype=np.float64)
    for query in queries:
        exact.extend(
            {"query_accession": query.accession, "reference_accession": accession}
            for accession in hashes.get(peak_hash(query), [])
        )
        # Vectorized precursor screening limits this audit to plausible copied spectra.
        positions = np.flatnonzero(
            np.abs(masses - query.precursor_mz) / masses * 1e6 <= 5.0
        )
        query_spectrum = query.spectrum("blind", blind=True)
        for index in positions:
            reference = library[index]
            value = scorer.pair(query_spectrum, reference.spectrum(reference.accession))
            if (
                float(value["score"]) >= 0.99
                and int(value["matches"]) >= 0.9 * len(query.mz)
                and int(value["matches"]) >= 0.9 * len(reference.mz)
            ):
                near.append(
                    {
                        "query_accession": query.accession,
                        "reference_accession": reference.accession,
                        "score": float(value["score"]),
                        "matched_peaks": int(value["matches"]),
                        "independence": "different_declared_laboratories_no_raw_scan_link",
                    }
                )
    return {
        "hash_policy": "exact_mz_float64_basepeak_normalized_round12",
        "near_rule": {
            "precursor_ppm": 5.0,
            "direct_cosine_tolerance_da": 0.005,
            "minimum_cosine": 0.99,
            "matched_fraction_each": 0.9,
        },
        "exact_cross_pairs": exact,
        "near_cross_pairs": near,
        "repeated_library_identities": {
            key: value for key, value in identities.items() if value > 1
        },
        "within_library_exact_groups": [
            values for values in hashes.values() if len(values) > 1
        ],
    }
