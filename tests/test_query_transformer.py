from app.services.query_transformer import QueryTransformer


def test_query_rewrite_extracts_code_like_entities():
    rewritten, entities = QueryTransformer().rewrite(
        "Why does auth/login.py raise JWTDecodeError in v1.2.3?"
    )

    assert "JWTDecodeError" in entities
    assert "v1.2.3" in entities
    assert rewritten.startswith("Why does")
    assert rewritten.count("JWTDecodeError") == 2
