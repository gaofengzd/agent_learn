from pydantic_settings import BaseSettings

class AppConfig(BaseSettings):
    api_key: str
    api_base_url: str
    llm_model: str
    log_level: str = "INFO"  # 给了默认值 "INFO"，代表选填，.env 里没写就用默认值

    # 自动从 .env 文件加载配置
    model_config = {"env_file": ".env"}

# 加载配置，自动校验类型，缺失会直接报错
# config = AppConfig()   # 这一行代码执行的时候，才是真正自动加载的时刻
# print(config.api_key)