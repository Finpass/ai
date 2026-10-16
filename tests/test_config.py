import pytest

from src.core.config import Settings


def test_production_rejects_default_key():
    with pytest.raises(ValueError, match="INTERNAL_API_KEY"):
        Settings(app_env="production")


def test_production_accepts_real_key():
    assert Settings(app_env="production", internal_api_key="s3cret").is_production
