import time

from MassFlow.cheminformatics import calculate_isotopic_envelope
from MassFlow.models import MolecularStructure


def generate_unique_smiles(n: int) -> list[str]:
    """Generate a list of n unique, valid SMILES strings."""
    smiles_list = []
    # Create unique chains by varying length and end groups
    # This ensures RDKit parses them validly and they have diverse exact masses
    groups = [
        "O",
        "N",
        "S",
        "F",
        "Cl",
        "C(=O)O",
        "C(=O)N",
        "C#N",
        "c1ccccc1",
        "C1CCCCC1",
    ]
    for i in range(n):
        chain_length = (i // len(groups)) + 1
        group = groups[i % len(groups)]
        smiles_list.append(f"{'C' * chain_length}{group}")
    return smiles_list


def measure_instantiation(
    smiles_list: list[str], disable_envelope: bool = False
) -> float:
    """Measure the time to instantiate MolecularStructure for a list of SMILES."""
    start = time.perf_counter()
    if disable_envelope:
        # Pass a dummy envelope to bypass the auto-calculation logic in the validator
        dummy_env = [(0.0, 1.0)]
        for s in smiles_list:
            MolecularStructure(smiles=s, isotopic_envelope=dummy_env)
    else:
        for s in smiles_list:
            MolecularStructure(smiles=s)
    end = time.perf_counter()
    return end - start


def main():
    print("=" * 60)
    print("MassFlow MolecularStructure Benchmark")
    print("=" * 60)

    n_molecules = 1000
    print(f"\n1. Generating {n_molecules} unique diverse SMILES strings...")
    smiles_list = generate_unique_smiles(n_molecules)

    # Pre-warm RDKit and Pyteomics to avoid initialization overhead in the first benchmark
    MolecularStructure(smiles="C")
    calculate_isotopic_envelope.cache_clear()

    print("\n2. Running Benchmark: Base Overhead (Isotopic Envelope Bypassed)")
    time_base = measure_instantiation(smiles_list, disable_envelope=True)
    avg_base = (time_base / n_molecules) * 1000
    print(f"   Total time: {time_base:.4f} seconds")
    print(f"   Avg latency: {avg_base:.4f} ms / molecule")

    print("\n3. Running Benchmark: Cold Cache (Calculating Isotopic Envelopes)")
    calculate_isotopic_envelope.cache_clear()
    time_cold = measure_instantiation(smiles_list, disable_envelope=False)
    avg_cold = (time_cold / n_molecules) * 1000
    print(f"   Total time: {time_cold:.4f} seconds")
    print(f"   Avg latency: {avg_cold:.4f} ms / molecule")

    print("\n4. Running Benchmark: Hot Cache (Repeat Calculations)")
    time_hot = measure_instantiation(smiles_list, disable_envelope=False)
    avg_hot = (time_hot / n_molecules) * 1000
    print(f"   Total time: {time_hot:.4f} seconds")
    print(f"   Avg latency: {avg_hot:.4f} ms / molecule")

    cache_info = calculate_isotopic_envelope.cache_info()
    print(f"\n   LRU Cache Status: {cache_info}")

    print("\n" + "=" * 60)
    print("Conclusion Report")
    print("=" * 60)

    overhead_cold = avg_cold - avg_base
    overhead_hot = avg_hot - avg_base

    print(
        f"Isotopic envelope COLD calculation overhead: ~{overhead_cold:.4f} ms / molecule"
    )
    print(
        f"Isotopic envelope HOT cache overhead:        ~{overhead_hot:.4f} ms / molecule"
    )

    if time_hot < time_cold * 0.2:
        print("\n[SUCCESS] The @lru_cache is highly effective!")
        print(f"          Repeat calculations are {time_cold / time_hot:.1f}x faster.")
    else:
        print("\n[WARNING] The @lru_cache might not be working as expected.")


if __name__ == "__main__":
    main()
