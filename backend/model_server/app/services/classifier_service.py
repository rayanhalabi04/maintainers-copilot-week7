from app.domain.classification import ClassifyRequest, ClassifyResponse
from app.infra.model_loader import load_model, load_model_card


class ClassifierService:
    def __init__(self) -> None:
        self.model = load_model()
        self.model_card = load_model_card()

    def classify(self, request: ClassifyRequest) -> ClassifyResponse:
        text = self._build_issue_text(request.title, request.body)

        predicted_label = self.model.predict([text])[0]

        probabilities = self.model.predict_proba([text])[0]
        class_names = list(self.model.classes_)

        probabilities_dict = {
            label: float(probability)
            for label, probability in zip(class_names, probabilities)
        }

        confidence = float(max(probabilities))

        return ClassifyResponse(
            label=str(predicted_label),
            confidence=confidence,
            probabilities=probabilities_dict,
            model_name=self.model_card.get("model_name", "tfidf_logreg_baseline"),
        )

    def _build_issue_text(self, title: str, body: str | None) -> str:
        body = body or ""
        return f"{title}\n\n{body}".strip()