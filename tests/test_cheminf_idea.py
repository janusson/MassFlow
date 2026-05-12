import re

COMMON_NEUTRAL_LOSSES = [
    (18.0106, {"O"}),
    (17.0265, {"N"}),
    (27.9949, {"O"}),
    (43.9898, {"O"}),
    (34.9956, {"S"}),
    (63.9619, {"S", "O"}),
    (78.9585, {"P", "O"}),
    (35.9767, {"Cl"}),
    (79.9262, {"Br"}),
    (20.0062, {"F"}),
]


def parse_elements_from_smiles(smiles: str) -> set:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return set()
    formula = rdMolDescriptors.CalcMolFormula(mol)
    matches = re.findall(r"([A-Z][a-z]*)(\d*)", formula)
    return {element for element, _ in matches}


def find_impossible_neutral_losses(
    mz_array, int_array, precursor_mz, smiles, tolerance=0.02, intensity_threshold=0.05
):
    atoms = parse_elements_from_smiles(smiles)
    if not atoms:
        return []

    max_int = max(int_array) if len(int_array) > 0 else 0
    if max_int == 0:
        return []

    impossible_losses = []

    # Consider only significant peaks
    for mz, intensity in zip(mz_array, int_array):
        if intensity / max_int < intensity_threshold:
            continue

        nl = precursor_mz - mz
        if nl <= 0:
            continue

        for exact_mass, required_atoms in COMMON_NEUTRAL_LOSSES:
            if abs(nl - exact_mass) <= tolerance:
                if not required_atoms.issubset(atoms):
                    impossible_losses.append((nl, exact_mass, required_atoms))

    return impossible_losses


print(
    find_impossible_neutral_losses([100.0, 150.0], [100, 100], 118.01, "C")
)  # Neutral loss = 18.01. Target = C. Output should show impossible H2O loss.
print(
    find_impossible_neutral_losses([100.0, 150.0], [100, 100], 118.01, "CCO")
)  # Target = CCO. Output should be empty since O is present.
