from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "notus-label-agent"
    app_version: str = "0.1.0"
    debug: bool = False

    database_url: str
    database_url_sync: str

    redis_url: str = "redis://redis:6379/0"
    cache_ttl_seconds: int = 300

    tesseract_cmd: str = "/usr/bin/tesseract"
    paddle_use_gpu: bool = False

    uploads_dir: str = "uploads"
    max_upload_size_mb: int = 10

    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
