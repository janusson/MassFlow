import os
import numpy as np
from typing import List, Tuple
from matchms import Spectrum, calculate_scores
from matchms.similarity import CosineGreedy
from src.utils import get_logger

# Import spec2vec conditionally or catch error if not installed
try:
    from spec2vec import Spec2Vec
    SPEC2VEC_AVAILABLE = True
except ImportError:
    SPEC2VEC_AVAILABLE = False

logger = get_logger(__name__)

class Searcher:
    def __init__(self, config: dict):
        self.config = config
        self.model_config = config['models']
        self.selected_model = self.model_config['selected_model']

    def search(self, query_spectrum: Spectrum, library: List[Spectrum], n_best: int = 10) -> List[Tuple[Spectrum, float]]:
        """
        Searches the library for spectra similar to the query spectrum.
        """
        if self.selected_model == 'cosine':
            return self._search_cosine(query_spectrum, library, n_best)
        elif self.selected_model == 'spec2vec':
            return self._search_spec2vec(query_spectrum, library, n_best)
        else:
            raise ValueError(f"Unknown model selected: {self.selected_model}")

    def _search_cosine(self, query: Spectrum, library: List[Spectrum], n_best: int) -> List[Tuple[Spectrum, float]]:
        """
        Performs Cosine Greedy search.
        """
        cosine_params = self.model_config['cosine']
        similarity_measure = CosineGreedy(
            tolerance=cosine_params.get('tolerance', 0.1),
            mz_power=cosine_params.get('mz_power', 0.0),
            intensity_power=cosine_params.get('intensity_power', 1.0)
        )

        logger.info("Calculating cosine similarity...")
        scores = calculate_scores(
            references=library,
            queries=[query],
            similarity_function=similarity_measure
        )

        best_matches = scores.scores_by_query(query, name="CosineGreedy_score", sort=True)
        # best_matches is a list of (reference, score) tuples
        return best_matches[:n_best]

    def _search_spec2vec(self, query: Spectrum, library: List[Spectrum], n_best: int) -> List[Tuple[Spectrum, float]]:
        """
        Performs Spec2Vec search.
        """
        if not SPEC2VEC_AVAILABLE:
            raise ImportError("Spec2Vec is not installed.")

        spec2vec_params = self.model_config['spec2vec']
        model_file = os.path.join(self.model_config['model_path'], spec2vec_params['model_file'])

        if not os.path.exists(model_file):
             # For now, let's assume if model file is missing we can't run spec2vec
             # In a real scenario we might load a default model or download one
             raise FileNotFoundError(f"Spec2Vec model file not found: {model_file}")

        import gensim
        logger.info(f"Loading Spec2Vec model from {model_file}...")
        model = gensim.models.Word2Vec.load(model_file)

        similarity_measure = Spec2Vec(
            model=model,
            intensity_weighting_power=spec2vec_params.get('intensity_weighting_power', 0.5),
            allowed_missing_percentage=spec2vec_params.get('allowed_missing_percentage', 5.0)
        )

        logger.info("Calculating Spec2Vec similarity...")
        scores = calculate_scores(
            references=library,
            queries=[query],
            similarity_function=similarity_measure
        )

        best_matches = scores.scores_by_query(query, name="Spec2Vec", sort=True)
        return best_matches[:n_best]
