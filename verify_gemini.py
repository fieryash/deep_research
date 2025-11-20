import os
from deep_research.config import AppConfig
from deep_research.llm import init_models
from langchain_core.messages import HumanMessage

def test_gemini_init():
    # Mock environment variable if not present for testing purposes, 
    # but in real scenario it should be in .env
    if "GOOGLE_API_KEY" not in os.environ:
        print("WARNING: GOOGLE_API_KEY not found in env. Please ensure it is set in .env")
        # We can't really test without a key, but we can test the config loading part at least
    
    try:
        config = AppConfig()
        print("Config loaded successfully.")
        print(f"Researcher model: {config.models.researcher_model}")
        
        if "google" not in config.models.researcher_model:
             print("ERROR: Researcher model is not set to google.")
             return

        # Only try to init models if we have a key, otherwise it will raise ValueError
        if config.google_api_key:
            models = init_models(config)
            print("Models initialized successfully.")
            
            # Optional: Try a simple invocation if you want to be really sure
            # response = models["researcher"].invoke([HumanMessage(content="Hello")])
            # print(f"Response from Gemini: {response.content}")
        else:
             print("Skipping model initialization test due to missing API key.")

    except Exception as e:
        print(f"Verification failed: {e}")

if __name__ == "__main__":
    test_gemini_init()
