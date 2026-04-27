from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
import os
import traceback
import time

app = Flask(__name__)
CORS(app) 

# Setup the AI Client 
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

syncbot_persona = """You are SyncBot, a highly specialized academic teaching assistant. Your sole purpose is to explain and discuss topics related exclusively to Unit 4: Synchronization in Digital Communications. 

You are strictly limited to the following core topics and their direct sub-topics:
1. Synchronization (General Concepts)
2. Phase Locked Loop (PLL)
3. Suppressed Carrier Loops
4. Costas Loop
5. Symbol/Bit Synchronization
6. Open-loop Synchronizers
7. Closed-loop Synchronizers (Early/Late Gate)
8. Frame Synchronization
9. Network/Transmitter Synchronization

STRICT RULES FOR INTERACTION:
- CONCISENESS: Your responses MUST be brief, punchy, and easy to read quickly. Avoid long paragraphs. Use short bullet points to break down concepts. Provide a high-level summary first, and only provide deep details if the user explicitly asks for them.
- GREETINGS: If the user sends a standard greeting (e.g., "hi", "hello", "hey", "good morning"), be polite! Reply with a brief, friendly greeting and ask how you can help them with Unit 4 Synchronization today. Do NOT use the refusal message for simple greetings.
- OFF-TOPIC REFUSAL: If a user asks a question about ANY topic outside the core synchronization list above, you MUST refuse to answer. Use this exact format: "I am SyncBot, and I am programmed to strictly discuss Unit 4 Synchronization topics. I cannot answer questions outside of this scope."
"""

# A helper function to safely call the API and retry if Google is busy
def retry_api_call(gemini_history, retries=3, delay=2):
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=gemini_history,
                config=types.GenerateContentConfig(system_instruction=syncbot_persona)
            )
            return response.text
        except Exception as e:
            error_str = str(e)
            # If it's a 503 error, wait and try again
            if "503" in error_str and attempt < retries - 1:
                print(f"⚠️ Google API busy. Retrying in {delay} seconds... (Attempt {attempt + 1}/{retries})")
                time.sleep(delay)
            else:
                # If it's not a 503, or we ran out of retries, actually crash
                raise e

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get("message")
        frontend_history = data.get("history", []) 
        
        # 1. Safely translate frontend history into Google's strict format
        gemini_history = []
        for m in frontend_history:
            # If the frontend says 'bot' or 'assistant', force it to be 'model'
            safe_role = "user" if m["role"] == "user" else "model"
            gemini_history.append(
                types.Content(role=safe_role, parts=[types.Part.from_text(text=m["content"])])
            )
        
        # 2. Add the user's newest message
        gemini_history.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_message)])
        )

        # 3. Call the API using our retry logic
        reply_text = retry_api_call(gemini_history)
        
        return jsonify({"reply": reply_text})
    
    except Exception as e:
        print("\n=== BACKEND CRASHED ===")
        traceback.print_exc() 
        print("=======================\n")
        return jsonify({"error": str(e)}), 500