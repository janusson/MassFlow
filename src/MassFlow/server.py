"""
Language Server Protocol (LSP) implementation for MassFlow.

This server interfaces with editors (like Zed) to provide real-time
mass spectrometry insights, structural validation, and hover context
directly within text, CSV, or Markdown files.
"""

import logging
import re
from typing import Optional

from lsprotocol.types import (
    TEXT_DOCUMENT_DID_CHANGE,
    TEXT_DOCUMENT_DID_OPEN,
    TEXT_DOCUMENT_HOVER,
    Diagnostic,
    DiagnosticSeverity,
    Hover,
    HoverParams,
    MarkupContent,
    MarkupKind,
    Position,
    Range,
)
from pygls.lsp.server import LanguageServer
from rdkit import Chem

logger = logging.getLogger("massflow_lsp")
server = LanguageServer("massflow-lsp", "v1.0")

# --- Regular Expressions ---
# Matches plausible scan IDs like 'Scan_001', 'scan_1234', 'query_05'
SCAN_ID_RE = re.compile(r"\b(?:Scan|scan|query)_[a-zA-Z0-9]+\b")

# Matches plausible chemical formulas (e.g., C8H10N4O2)
FORMULA_RE = re.compile(r"\b([A-Z][a-z]?\d*)+\b")

# Matches plausible SMILES (very basic heuristic to trigger RDKit validation)
# Looks for chains of C, N, O, S, rings (1-9), and brackets
SMILES_RE = re.compile(r"\b([CNOPSFclbrI\=\#\(\)\[\]\@\+\-\.\d]{4,})\b")

# --- In-Memory Trie for Editor Context ---
# In a production environment, this is loaded from the SQLite library at startup.
memory_trie = None
MOCK_SCAN_DB = {
    "Scan_001": {
        "name": "Caffeine",
        "mz": 195.088,
        "score": 0.98,
        "adduct": "[M+H]+",
    },
    "query_42": {
        "name": "Aspirin",
        "mz": 181.050,
        "score": 0.91,
        "adduct": "[M+H]+",
    },
}

VALID_ELEMENTS = {
    "C",
    "H",
    "N",
    "O",
    "P",
    "S",
    "F",
    "Cl",
    "Br",
    "I",
    "Na",
    "K",
    "Ca",
    "Fe",
    "Mg",
    "Mn",
    "Zn",
    "Cu",
}


def is_plausible_formula(text: str) -> bool:
    """Check if a matched string is actually a formula, not just a capitalized word."""
    # Reject plain words (e.g., "Hello", "The", "A")
    if text.isalpha() and text.istitle():
        return False
    if text.isalpha() and text.isupper() and len(text) < 3:
        return False

    elements = re.findall(r"[A-Z][a-z]?", text)
    if not elements:
        return False

    return all(e in VALID_ELEMENTS for e in elements)


def check_smiles_validity(smiles: str) -> Optional[str]:
    """Returns an error message if SMILES is invalid, else None."""
    # 1. RDKit Semantic validation
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return "Invalid SMILES syntax."

    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        return f"SMILES valency/sanitization error: {str(e)}"
    return None


@server.feature(TEXT_DOCUMENT_DID_OPEN)
@server.feature(TEXT_DOCUMENT_DID_CHANGE)
def validate_document(ls: LanguageServer, params):
    """
    Scans the document for structural artifacts (Formulas, SMILES)
    and publishes diagnostics (squiggly lines) for invalid ones.
    """
    doc = ls.workspace.get_document(params.text_document.uri)
    diagnostics = []

    for line_num, line in enumerate(doc.lines):
        # 1. Validate Formulas
        for match in FORMULA_RE.finditer(line):
            text = match.group(0)
            if not is_plausible_formula(text):
                continue

            # Basic validation: ensure no weird elements slipped through
            elements = re.findall(r"[A-Z][a-z]?", text)
            invalid_elements = [e for e in elements if e not in VALID_ELEMENTS]

            if invalid_elements:
                diagnostics.append(
                    Diagnostic(
                        range=Range(
                            start=Position(line=line_num, character=match.start()),
                            end=Position(line=line_num, character=match.end()),
                        ),
                        message=f"Invalid chemical formula. Unknown elements: {', '.join(invalid_elements)}",
                        severity=DiagnosticSeverity.Warning,
                        source="MassFlow",
                    )
                )

        # 2. Validate SMILES (Heuristic matching first)
        for match in SMILES_RE.finditer(line):
            text = match.group(0)

            # Skip numbers or simple words caught by the regex
            if text.isdigit() or (text.isalpha() and text.islower()):
                continue

            error = check_smiles_validity(text)
            if error:
                diagnostics.append(
                    Diagnostic(
                        range=Range(
                            start=Position(line=line_num, character=match.start()),
                            end=Position(line=line_num, character=match.end()),
                        ),
                        message=f"MassFlow: {error}",
                        severity=DiagnosticSeverity.Error,
                        source="MassFlow-RDKit",
                    )
                )

    ls.publish_diagnostics(doc.uri, diagnostics)


@server.feature(TEXT_DOCUMENT_HOVER)
def hover(ls: LanguageServer, params: HoverParams) -> Optional[Hover]:
    """
    Provides tooltips when hovering over scan IDs, formulas, or SMILES.
    """
    doc = ls.workspace.get_document(params.text_document.uri)
    word = doc.word_at_position(params.position)

    if not word:
        return None

    # 1. Check if it's a known Scan ID
    info = None
    if word in MOCK_SCAN_DB:
        info = MOCK_SCAN_DB[word]

    if info:
        content = (
            f"### 🔬 MassFlow Annotation\n"
            f"**Scan ID:** `{word}`\n\n"
            f"- **Best Match:** {info['name']}\n"
            f"- **Precursor *m/z*:** {info['mz']:.4f} {info['adduct']}\n"
            f"- **Similarity Score:** {info['score']:.2f}\n"
        )
        return Hover(contents=MarkupContent(kind=MarkupKind.Markdown, value=content))

    # 2. Check if it's a valid SMILES
    if SMILES_RE.match(word) and check_smiles_validity(word) is None:
        mol = Chem.MolFromSmiles(word)
        exact_mass = Chem.rdMolDescriptors.CalcExactMolWt(mol)
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        content = (
            f"### 🧪 Molecular Structure\n"
            f"**SMILES:** `{word}`\n\n"
            f"- **Formula:** {formula}\n"
            f"- **Exact Mass:** {exact_mass:.4f} Da\n"
        )
        return Hover(contents=MarkupContent(kind=MarkupKind.Markdown, value=content))

    # 3. Check if it's a valid Formula
    if FORMULA_RE.match(word) and is_plausible_formula(word):
        content = f"### 📊 Chemical Formula\n`{word}` appears to be a valid compositional formula."
        return Hover(contents=MarkupContent(kind=MarkupKind.Markdown, value=content))

    return None


if __name__ == "__main__":
    # Start the server via standard I/O (required for LSP)
    server.start_io()
