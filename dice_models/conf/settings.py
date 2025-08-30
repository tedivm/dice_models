from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project_name: str = "dice_models"
    debug: bool = False
