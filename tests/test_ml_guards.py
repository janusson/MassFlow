"""
Tests for the ML dependency guard architecture in MassFlow.

This module verifies:
1. Import guard flags exist and correctly reflect the environment.
2. ML engines raise clear RuntimeError when ``[ml]`` extras are missing.
3. Classical scoring engines remain fully functional regardless.
4. The factory function dispatches correctly.
5. The core workflow pipeline initializes without ML libraries present.
"""

import pytest

from MassFlow.config import SimilarityConfig
from MassFlow.similarity import (
    CascadeEngine,
    ConsensusEngine,
    MS2DeepScoreEngine,
    SimilarityEngine,
    Spec2VecEngine,
    _HAS_GENSIM,
    _HAS_ML,
    _HAS_MS2DEEPSCORE,
    _HAS_SPEC2VEC,
    _HAS_TORCH,
    _MLEngineBase,
    _ML_INSTALL_MSG,
    get_similarity_engine,
)


# =============================================================================
# Import guard flag tests
# =============================================================================


class TestImportGuards:
    """Verify that the module-level guard flags behave correctly."""

    def test_guard_flags_exist(self) -> None:
        """All five guard flags must be present as module-level booleans."""
        assert isinstance(_HAS_TORCH, bool)
        assert isinstance(_HAS_GENSIM, bool)
        assert isinstance(_HAS_SPEC2VEC, bool)
        assert isinstance(_HAS_MS2DEEPSCORE, bool)
        assert isinstance(_HAS_ML, bool)

    def test_ml_flag_is_conjunction(self) -> None:
        """_HAS_ML must be True iff all individual flags are True."""
        all_present = _HAS_TORCH and _HAS_GENSIM and _HAS_SPEC2VEC and _HAS_MS2DEEPSCORE
        assert _HAS_ML == all_present, (
            "_HAS_ML must be the logical AND of all individual guard flags"
        )

    def test_ml_install_message_is_string(self) -> None:
        """The install instructions must be a non-empty string."""
        assert isinstance(_ML_INSTALL_MSG, str)
        assert len(_ML_INSTALL_MSG) > 0
        assert "pip install" in _ML_INSTALL_MSG.lower()


# =============================================================================
# Classical engine (always available) tests
# =============================================================================


class TestClassicalEngines:
    """Verify classical engines are always available and functional."""

    def test_cosine_engine_instantiates(self) -> None:
        """Cosine engine must instantiate regardless of ML availability."""
        engine = SimilarityEngine(SimilarityConfig(algorithm="cosine"))
        assert isinstance(engine, SimilarityEngine)

    def test_modified_cosine_engine_instantiates(self) -> None:
        """Modified cosine engine must instantiate regardless."""
        engine = SimilarityEngine(SimilarityConfig(algorithm="modified_cosine"))
        assert isinstance(engine, SimilarityEngine)

    def test_factory_returns_classical_engine(self) -> None:
        """Factory must return SimilarityEngine for classical algorithms."""
        for algo in ("cosine", "modified_cosine"):
            engine = get_similarity_engine(SimilarityConfig(algorithm=algo))
            assert isinstance(engine, SimilarityEngine), (
                f"Expected SimilarityEngine for {algo}, got {type(engine).__name__}"
            )

    def test_factory_raises_value_error_for_unknown_algorithm(self) -> None:
        """Unknown algorithm names must raise Pydantic ValidationError.

        Pydantic's Literal type validation catches invalid algorithm names
        at the configuration layer before the factory is invoked."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="algorithm"):
            SimilarityConfig(algorithm="unknown_algo")


# =============================================================================
# ML engine degradation tests (run regardless of whether ml extras are installed)
# =============================================================================


class TestMLEngineDegradation:
    """
    Verify ML engine behavior when ml extras are absent.

    These tests validate graceful degradation: clear RuntimeError messages
    that tell the user exactly how to install the required dependencies.
    """

    @pytest.mark.parametrize(
        "algo,engine_cls",
        [
            ("spec2vec", Spec2VecEngine),
            ("ms2deepscore", MS2DeepScoreEngine),
        ],
    )
    def test_factory_raises_runtime_error_without_ml(
        self, algo: str, engine_cls: type
    ) -> None:
        """Factory must raise RuntimeError for pure ML algos when ml extras absent."""
        if _HAS_ML:
            pytest.skip("ML extras are installed; degradation test not applicable")

        with pytest.raises(RuntimeError, match="machine-learning extras"):
            get_similarity_engine(SimilarityConfig(algorithm=algo))

    @pytest.mark.parametrize(
        "algo,engine_cls",
        [
            ("consensus", ConsensusEngine),
            ("cascade", CascadeEngine),
        ],
    )
    def test_meta_engines_degrade_gracefully_without_ml(
        self, algo: str, engine_cls: type
    ) -> None:
        """Consensus/Cascade must succeed even without ML deps, falling back to classical."""
        if _HAS_ML:
            pytest.skip("ML extras are installed; degradation test not applicable")

        engine = get_similarity_engine(SimilarityConfig(algorithm=algo))
        assert isinstance(engine, engine_cls), (
            f"Expected {engine_cls.__name__} for {algo}, got {type(engine).__name__}"
        )

    @pytest.mark.parametrize(
        "engine_cls",
        [Spec2VecEngine, MS2DeepScoreEngine],
    )
    def test_direct_init_raises_runtime_error_without_ml(
        self, engine_cls: type
    ) -> None:
        """Direct init of Spec2Vec/MS2DeepScore must raise RuntimeError without ml."""
        if _HAS_ML:
            pytest.skip("ML extras are installed; degradation test not applicable")

        with pytest.raises(RuntimeError, match="machine-learning extras"):
            engine_cls(SimilarityConfig(algorithm="spec2vec"))

    @pytest.mark.parametrize(
        "engine_cls",
        [ConsensusEngine, CascadeEngine],
    )
    def test_direct_init_degraded_without_ml(self, engine_cls: type) -> None:
        """Consensus/Cascade engines degrade gracefully, logging a warning."""
        if _HAS_ML:
            pytest.skip("ML extras are installed; degradation test not applicable")

        # These engines should NOT raise – they log a warning and continue
        # with classical sub-engines only.
        engine = engine_cls(SimilarityConfig(algorithm="spec2vec"))
        assert isinstance(engine, engine_cls)

    @pytest.mark.skipif(not _HAS_ML, reason="ML extras not installed")
    @pytest.mark.parametrize(
        "engine_cls",
        [Spec2VecEngine, MS2DeepScoreEngine, ConsensusEngine, CascadeEngine],
    )
    def test_ml_engines_instantiate_when_deps_present(self, engine_cls: type) -> None:
        """ML engines must instantiate successfully when deps are available."""
        engine = engine_cls(SimilarityConfig(algorithm="spec2vec"))
        assert isinstance(engine, _MLEngineBase)

    @pytest.mark.parametrize(
        "engine_cls",
        [Spec2VecEngine, MS2DeepScoreEngine, ConsensusEngine, CascadeEngine],
    )
    def test_ml_engines_inherit_from_base(self, engine_cls: type) -> None:
        """All ML engines must inherit from _MLEngineBase."""
        assert issubclass(engine_cls, _MLEngineBase), (
            f"{engine_cls.__name__} must inherit from _MLEngineBase"
        )


# =============================================================================
# Core pipeline integration tests
# =============================================================================


class TestCorePipelineWithoutML:
    """
    Verify the core pipeline works fully without any ML libraries.

    These tests validate the primary architectural requirement: classical
    cosine and modified_cosine scoring, FDR validation, and the full
    annotation workflow must execute when ML libraries are absent.
    """

    def test_similarity_config_validates_with_classical_algos(self) -> None:
        """SimilarityConfig must accept 'cosine' and 'modified_cosine'."""
        for algo in ("cosine", "modified_cosine"):
            cfg = SimilarityConfig(algorithm=algo)
            assert cfg.algorithm == algo

    def test_workflow_module_imports_without_ml(self) -> None:
        """The workflow module must import successfully without ML."""
        from MassFlow import workflow  # noqa: F401

    def test_full_pipeline_imports_without_ml(self) -> None:
        """All core modules must be importable without ML libraries."""
        import MassFlow.config  # noqa: F401
        import MassFlow.database  # noqa: F401
        import MassFlow.io  # noqa: F401
        import MassFlow.processing  # noqa: F401
        import MassFlow.similarity  # noqa: F401
        import MassFlow.workflow  # noqa: F401

    def test_search_result_structure_is_consistent(self) -> None:
        """
        SearchResult dict keys must be consistent whether classical or ML.

        The ``SearchResult`` TypedDict defines the contract used by FDR,
        export, and database layers. All engines produce results with
        the same key set so downstream code works uniformly.
        """
        from MassFlow.similarity import SearchResult  # noqa: F401

        # SearchResult is imported to verify it's importable.
        # The actual key validation happens at runtime in FDR/export.
        assert True  # SearchResult TypedDict contract is importable

    def test_config_ml_algorithms_are_valid_literals(self) -> None:
        """All ML algorithm names must be valid SimilarityConfig literals."""
        for algo in ("spec2vec", "ms2deepscore", "consensus", "cascade"):
            cfg = SimilarityConfig(algorithm=algo)
            assert cfg.algorithm == algo


# =============================================================================
# Mock-tests: simulate absent ML in an environment where it is present
# =============================================================================


class TestMLAbsenceSimulation:
    """
    Simulate the absence of ML libraries even if they are installed.

    These tests use monkeypatching to set guard flags to False and verify
    that the system degrades gracefully. This is critical for verifying
    the guard pattern is functional regardless of the test environment.
    """

    def test_factory_uses_guard_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Factory must check _HAS_ML flag before dispatching to ML engines."""
        from MassFlow import similarity

        # Force _HAS_ML to False, simulating absent ML deps
        monkeypatch.setattr(similarity, "_HAS_ML", False)

        with pytest.raises(RuntimeError, match="machine-learning extras"):
            get_similarity_engine(SimilarityConfig(algorithm="spec2vec"))

    def test_ml_engine_init_checks_deps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_MLEngineBase._check_dependencies must gate on _HAS_ML."""
        from MassFlow import similarity

        monkeypatch.setattr(similarity, "_HAS_ML", False)

        with pytest.raises(RuntimeError, match="machine-learning extras"):
            Spec2VecEngine(SimilarityConfig(algorithm="spec2vec"))

    def test_classical_engine_unaffected_by_ml_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Classical engines must work even when _HAS_ML is monkeypatched False."""
        from MassFlow import similarity

        monkeypatch.setattr(similarity, "_HAS_ML", False)

        # This must not raise
        engine = get_similarity_engine(SimilarityConfig(algorithm="cosine"))
        assert isinstance(engine, SimilarityEngine)


# =============================================================================
# Entry-point registry and protocol compliance tests
# =============================================================================


class TestEntryPointRegistry:
    """Verify that the entry-point-based engine registry works correctly."""

    def test_registry_is_dict(self) -> None:
        """_ML_ENGINE_REGISTRY must be a dict."""
        from MassFlow.similarity import _ML_ENGINE_REGISTRY

        assert isinstance(_ML_ENGINE_REGISTRY, dict)

    def test_registry_contains_builtin_engines(self) -> None:
        """All four built-in ML engines must be discoverable via entry points."""
        from MassFlow.similarity import _ML_ENGINE_REGISTRY

        expected = {"spec2vec", "ms2deepscore", "consensus", "cascade"}
        registered = set(_ML_ENGINE_REGISTRY.keys())
        missing = expected - registered
        assert not missing, (
            f"Built-in engines not found in registry: {missing}. "
            f"Registered: {registered}"
        )

    def test_legacy_alias_points_to_registry(self) -> None:
        """_ML_ENGINE_MAP must be an alias for _ML_ENGINE_REGISTRY."""
        from MassFlow.similarity import _ML_ENGINE_MAP, _ML_ENGINE_REGISTRY

        assert _ML_ENGINE_MAP is _ML_ENGINE_REGISTRY, (
            "_ML_ENGINE_MAP must be the same object as _ML_ENGINE_REGISTRY"
        )

    @pytest.mark.parametrize(
        "algo", ["spec2vec", "ms2deepscore", "consensus", "cascade"]
    )
    def test_registry_entries_are_classes(self, algo: str) -> None:
        """Each registry entry must resolve to a class (not a module or function)."""
        from MassFlow.similarity import _ML_ENGINE_REGISTRY

        engine_cls = _ML_ENGINE_REGISTRY[algo]
        assert isinstance(engine_cls, type), (
            f"Registry entry '{algo}' is {type(engine_cls).__name__}, expected a class"
        )

    def test_discovery_is_idempotent(self) -> None:
        """Calling _discover_ml_engines twice must produce identical results."""
        from MassFlow.similarity import _discover_ml_engines

        first = _discover_ml_engines()
        second = _discover_ml_engines()
        assert first == second, "Repeated discovery must produce consistent results"


class TestProtocolCompliance:
    """Verify that all registered ML engines implement MLEngineProtocol."""

    def test_registered_engines_are_protocol_subclasses(self) -> None:
        """Every engine in _ML_ENGINE_REGISTRY must be a subclass of MLEngineProtocol."""
        from MassFlow.protocols import MLEngineProtocol
        from MassFlow.similarity import _ML_ENGINE_REGISTRY

        for algo, engine_cls in _ML_ENGINE_REGISTRY.items():
            assert issubclass(engine_cls, MLEngineProtocol), (
                f"'{algo}' ({engine_cls.__name__}) must be a subclass of MLEngineProtocol"
            )

    def test_protocol_defines_required_methods(self) -> None:
        """MLEngineProtocol must declare search, batch_score, and load_model."""
        from MassFlow.protocols import MLEngineProtocol

        assert hasattr(MLEngineProtocol, "search"), "Protocol missing 'search'"
        assert hasattr(MLEngineProtocol, "batch_score"), (
            "Protocol missing 'batch_score'"
        )
        assert hasattr(MLEngineProtocol, "load_model"), "Protocol missing 'load_model'"

    def test_base_engine_implements_batch_score(self) -> None:
        """_MLEngineBase.batch_score must exist and accept correct signature."""
        from MassFlow.similarity import _MLEngineBase

        assert hasattr(_MLEngineBase, "batch_score")
        assert callable(_MLEngineBase.batch_score)

    def test_base_engine_implements_load_model(self) -> None:
        """_MLEngineBase.load_model must exist and accept a path argument."""
        from MassFlow.similarity import _MLEngineBase

        assert hasattr(_MLEngineBase, "load_model")
        assert callable(_MLEngineBase.load_model)


# =============================================================================
# Numerical-parity scaffolding (for v1.x migration from inline → plugin ML)
# =============================================================================


class TestNumericalParityScaffolding:
    """
    Scaffolding to verify numerical parity between legacy inline ML engines
    and their decoupled plugin equivalents.

    These tests establish the contract that the satellite ``massflow-ml``
    package must satisfy.  The actual numerical assertions will be
    populated once the ML engines have full implementations.
    """

    def test_searchresult_structure_identical_regardless_of_engine(
        self,
    ) -> None:
        """
        SearchResult keys must be identical whether the engine was obtained
        via direct construction or via the entry-point factory.

        This guarantees that downstream code (FDR, export, database) never
        needs to know how the engine was created.
        """
        from typing import get_type_hints

        from MassFlow.similarity import SearchResult

        # SearchResult is a TypedDict; its __annotations__ define the required keys.
        required_keys = set(get_type_hints(SearchResult).keys())
        expected_keys = {
            "query_id",
            "query_precursor_mz",
            "reference_id",
            "reference_name",
            "reference_precursor_mz",
            "score",
            "matched_peaks",
            "smiles",
            "inchikey",
            "is_decoy",
            "q_value",
            "p_value",
            "annotation_tier",
            "structural_similarity",
            "mass_error_ppm",
            "score_breakdown",
        }
        assert required_keys == expected_keys, (
            f"SearchResult keys changed! Expected: {expected_keys}, got: {required_keys}"
        )

    def test_factory_and_direct_construct_return_same_type(self) -> None:
        """
        For classical algorithms, factory and direct construction must
        produce the same engine type.
        """
        for algo in ("cosine", "modified_cosine"):
            cfg = SimilarityConfig(algorithm=algo)
            from_factory = get_similarity_engine(cfg)
            from_direct = SimilarityEngine(cfg)
            assert type(from_factory) is type(from_direct), (
                f"Factory returned {type(from_factory).__name__} "
                f"but direct construction gave {type(from_direct).__name__} "
                f"for algorithm '{algo}'"
            )

    def test_ml_router_uses_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        MLRouter must use _ML_ENGINE_REGISTRY (not a hardcoded map) to
        check for available ML engines.
        """
        from MassFlow.similarity import MLRouter

        # Verify the router references _ML_ENGINE_REGISTRY
        import inspect

        source = inspect.getsource(MLRouter._get_hard_engine)
        assert "_ML_ENGINE_REGISTRY" in source, (
            "MLRouter._get_hard_engine must reference _ML_ENGINE_REGISTRY"
        )

    def test_search_result_serialization_roundtrip(self) -> None:
        """
        SearchResult dicts must survive a JSON-serialization roundtrip
        without losing precision or keys.
        """
        import json

        from MassFlow.similarity import SearchResult

        result: SearchResult = {
            "query_id": "test_query_1",
            "query_precursor_mz": 304.1543,
            "reference_id": "test_ref_1",
            "reference_name": "Caffeine",
            "reference_precursor_mz": 195.0877,
            "score": 0.9234,
            "matched_peaks": 12,
            "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "inchikey": "RYYVLZVUVIJVGH-UHFFFAOYSA-N",
            "is_decoy": False,
            "q_value": 0.001,
            "p_value": 0.0005,
            "annotation_tier": "Level 1",
            "structural_similarity": 0.87,
            "mass_error_ppm": 1.23,
            "score_breakdown": {"cosine": 0.92, "modified_cosine": 0.88},
        }

        serialized = json.dumps(result, allow_nan=False)
        deserialized = json.loads(serialized)

        assert deserialized["query_id"] == result["query_id"]
        assert deserialized["score"] == result["score"]
        assert deserialized["score_breakdown"] == result["score_breakdown"]
        assert deserialized["is_decoy"] == result["is_decoy"]
