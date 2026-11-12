from config.settings import settings

print("OPENAI_API_KEY:", settings.OPENAI_API_KEY[:10] + "******")
print("OPENAI_BASE_URL:", settings.OPENAI_BASE_URL)
print("MODEL_NAME:", settings.MODEL_NAME)
