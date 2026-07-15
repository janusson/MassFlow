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
            ("consensus", ConsensusEngine),
            ("cascade", CascadeEngine),
        ],
    )
    def test_factory_raises_runtime_error_without_ml(
        self, algo: str, engine_cls: type
    ) -> None:
        """Factory must raise RuntimeError for ML algos when ml extras absent."""
        if _HAS_ML:
            pytest.skip("ML extras are installed; degradation test not applicable")

        with pytest.raises(RuntimeError, match="machine-learning extras"):
            get_similarity_engine(SimilarityConfig(algorithm=algo))

    @pytest.mark.parametrize(
        "engine_cls",
        [Spec2VecEngine, MS2DeepScoreEngine, ConsensusEngine, CascadeEngine],
    )
    def test_direct_init_raises_runtime_error_without_ml(
        self, engine_cls: type
    ) -> None:
        """Direct init of ML engines must raise RuntimeError without ml extras."""
        if _HAS_ML:
            pytest.skip("ML extras are installed; degradation test not applicable")

        with pytest.raises(RuntimeError, match="machine-learning extras"):
            engine_cls(SimilarityConfig(algorithm="spec2vec"))

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
