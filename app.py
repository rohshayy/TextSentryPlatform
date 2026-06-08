from fastapi import FastAPI
from orchestrator import PlatformOrchestrator

# 1. Turn on the FastAPI framework
app = FastAPI()

# 2. Wake up your trained AI platform (loads weights, config, and vocab)
orchestrator = PlatformOrchestrator()


# 3. Create the basic landing page route
@app.get("/")
def home_page():
    return {"message": "Welcome! The AI TextSentry Web Server is live."}


# 4. Create the prediction route (The Function Mapping)
@app.get("/predict")
def run_prediction(text_input: str):
    # Pass the text coming from the web browser straight into your AI engine
    route, confidence = orchestrator.process_query(text_input)

    # Return the final calculated answers as a dictionary
    return {
        "text_received": text_input,
        "selected_route": route.upper(),
        "confidence_score": float(confidence)
    }