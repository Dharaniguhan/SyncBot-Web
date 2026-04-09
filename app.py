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

syncbot_persona = """You are SyncBot, a friendly and helpful assistant specializing strictly in Synchronization in Digital Communication. 
Your primary goals are to teach, explain, and quiz users on topics like Phase-Locked Loops (PLL), Frame Synchronization, Carrier Synchronization, Symbol Timing Recovery, etc.

CRITICAL INSTRUCTION ON TONE AND LENGTH: 
1. Keep your answers SHORT, concise, and punchy. Avoid giant walls of text.
2. Keep the language SIMPLE. Break down complex math or theories into plain English. 
3. Use real-world analogies whenever possible. Imagine you are explaining this to a second-year ECE student who is hearing about these concepts for the very first time.

CRITICAL RULE ON TOPIC: If a user asks a question completely unrelated to electronics, telecommunications, or synchronization, you MUST politely decline to answer and steer the conversation back to your area of expertise."""

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