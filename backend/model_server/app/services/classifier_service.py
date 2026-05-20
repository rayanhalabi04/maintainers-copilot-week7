from app.domain.classification import ClassifyRequest, ClassifyResponse
from app.infra.model_loader import load_model, load_model_card


LABEL_ID_TO_NAME = {
    "0": "bug",
    "1": "feature",
    "2": "docs",
    "3": "question",
}


class ClassifierService:
    def __init__(self) -> None:
        self.model = load_model()
        self.model_card = load_model_card()

    def classify(self, request: ClassifyRequest) -> ClassifyResponse:
        text = self._build_issue_text(request.title, request.body)

        raw_predicted_label = self.model.predict([text])[0]

        probabilities = self.model.predict_proba([text])[0]
        class_names = list(self.model.classes_)

        probabilities_dict = {
            self._decode_label(label): float(probability)
            for label, probability in zip(class_names, probabilities)
        }

        readable_label = self._decode_label(raw_predicted_label)
        confidence = float(max(probabilities))

        return ClassifyResponse(
            label=readable_label,
            confidence=confidence,
            probabilities=probabilities_dict,
            model_name=self.model_card.get("model_name", "tfidf_logreg_baseline"),
        )

    def _build_issue_text(self, title: str, body: str | None) -> str:
        body = body or ""
        return f"{title}\n\n{body}".strip()

    def _decode_label(self, label: object) -> str:
        label_as_string = str(label)
        return LABEL_ID_TO_NAME.get(label_as_string, label_as_string)