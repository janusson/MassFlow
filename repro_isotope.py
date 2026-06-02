from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from typing import DefaultDict

try:
    from pyteomics import mass as pmass
except ImportError:
    pmass = None


def debug_smiles(smiles):
    print(f"\nDebugging SMILES: {smiles}")
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        print("Invalid SMILES")
        return
    print(f"Formula: {rdMolDescriptors.CalcMolFormula(mol)}")
    print(f"H count: {sum(atom.GetTotalNumHs() for atom in mol.GetAtoms())}")

    # Check for isotopes
    has_isotope = any(atom.GetIsotope() > 0 for atom in mol.GetAtoms())
    print(f"Has isotope: {has_isotope}")

    # Manual formula construction logic
    mol_no_iso = Chem.Mol(mol)
    for atom in mol_no_iso.GetAtoms():
        atom.SetIsotope(0)
    base_formula = rdMolDescriptors.CalcMolFormula(mol_no_iso)
    print(f"Base formula (no iso): {base_formula}")

    if pmass:
        import re
        from collections import defaultdict

        base_matches = re.findall(r"([A-Z][a-z]*)(\d*)", base_formula)
        base_counts = {
            element: int(count) if count else 1 for element, count in base_matches
        }
        counts: DefaultDict[str, DefaultDict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            isotope = atom.GetIsotope()
            counts[symbol][isotope] += 1

        parts = []
        for symbol in sorted(base_counts.keys()):
            total = base_counts[symbol]
            if symbol in counts:
                isotopic_items = [
                    (iso, counts[symbol][iso])
                    for iso in sorted(counts[symbol].keys())
                    if iso > 0
                ]
                for iso, cnt in isotopic_items:
                    parts.append(f"{symbol}[{iso}]{cnt}")
                non_iso_count = counts[symbol].get(0, 0)
                if non_iso_count > 0:
                    parts.append(f"{symbol}{non_iso_count}")
                sum_counts = non_iso_count + sum(cnt for _, cnt in isotopic_items)
                if sum_counts < total:
                    rem = total - sum_counts
                    parts.append(f"{symbol}{rem}")
            else:
                parts.append(f"{symbol}{total}")
        formula_py = "".join(parts)
        print(f"Pyteomics formula: {formula_py}")
        try:
            mono_mass = pmass.calculate_mass(formula=formula_py)
            print(f"Pyteomics mass: {mono_mass}")
        except Exception as e:
            print(f"Pyteomics error: {e}")


debug_smiles("[13C]1=[13C][13C]=[13C][13C]=[13C]1")
debug_smiles("[13c]1[13c][13c][13c][13c][13c]1")
debug_smiles("c1ccccc1")
debug_smiles("[13CH]1=[13CH][13CH]=[13CH][13CH]=[13CH]1")
