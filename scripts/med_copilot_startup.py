"""Initialize patient embeddings before importing retrieval's FAISS dependencies."""

from sentence_transformers import SentenceTransformer

from scripts.med_copilot_config import ExperimentConfig


class PatientRetrievalStartup:
    """Preload only a requested encoder, retaining errors for dependent variants.

    Construct this before importing med_copilot_retrieval on Apple Silicon.
    main() reports configuration errors when it validates the run settings.
    """

    def __init__(self, config_path):
        self.experiment, self.encoder, self.encoder_error = None, None, None
        try:
            self.experiment = ExperimentConfig(config_path)
        except (OSError, ValueError):
            return
        if self.experiment.needs_patient_retrieval:
            try:
                self.encoder = SentenceTransformer(self.experiment.patient_embedding_model)
            except Exception as exc:
                self.encoder_error = exc

    def encoder_for(self, experiment):
        """Return the matching preloaded encoder or fail before mislabelling a cache."""
        if (self.experiment is None or not self.experiment.needs_patient_retrieval
                or self.experiment.patient_embedding_model != experiment.patient_embedding_model):
            raise ValueError("Patient embedding settings changed after startup; restart the script")
        if self.encoder_error is not None:
            raise self.encoder_error
        return self.encoder
