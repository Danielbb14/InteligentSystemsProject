import os
import time
from furhat_remote_api import FurhatRemoteAPI
import google.generativeai as genai
from dotenv import load_dotenv
import requests

# --- CONFIGURATION ---
load_dotenv() 

FURHAT_IP = "localhost"
GEMINI_API_KEY = os.getenv("API_KEY")
QUESTIONS_TO_ASK = 5

USE_KEYBOARD = False 

# --- MOOD CLASSIFIER CONFIGURATION ---
# >>> SET THIS TO FALSE TO DISABLE THE API CALLS <<<
MOOD_CLASSIFIER_ENABLED = True 
HARDCODED_MOOD = 'neutral' # Default mood when disabled or API fails
MOOD_API_URL = "http://127.0.0.1:8000/mood" 

# --- SETUP ---
genai.configure(api_key=GEMINI_API_KEY)
furhat = FurhatRemoteAPI(FURHAT_IP)

SYSTEM_INSTRUCTION = """
You are an empathetic, multimodal study assistant built for the Furhat robot. 
Your role is to listen attentively to the student’s academic questions and struggles, and provide clear, supportive guidance. 
Focus on helping the student solve problems and understand concepts. 
Keep responses concise, conversational, and emotionally aware, staying strictly on-topic.
"""

MOOD_PROMPT_MAPS = {
    'angry': "The user seems ANGRY/FRUSTRATED. Your primary goal is to validate their feelings and de-escalate the situation before attempting to solve the problem. Use calm, patient, and very supportive language.",
    'happy': "The user appears HAPPY and engaged! Match their positive energy. Keep explanations clear but feel free to be slightly more enthusiastic and suggest challenging follow-up ideas.",
    'neutral': "The user is NEUTRAL. Maintain a focused, professional, and clear tone. Stick strictly to solving the problem or explaining the concept without excess emotional output.",
    'sad': "The user seems SAD or dejected. Your response should be highly empathetic. Use soft language and focus on building confidence. Offer simple words of encouragement before the explanation."
}


model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=SYSTEM_INSTRUCTION
)

chat_session = model.start_chat(history=[])

# --- HELPER FUNCTIONS ---

def get_user_mood():
    """
    Calls the FastAPI server with a simple GET request to retrieve the
    latest continuously processed mood state. Falls back to HARDCODED_MOOD if
    MOOD_CLASSIFIER_ENABLED is False or if the API call fails.
    """
    if not MOOD_CLASSIFIER_ENABLED:
        return HARDCODED_MOOD

    try:
        response = requests.get(MOOD_API_URL, timeout=3)
        
        if response.status_code == 200:
            data = response.json()
            detected_mood = data.get('emotion', HARDCODED_MOOD)
            message = data.get('message', '')
            
            if 'No face detected' in message or detected_mood == HARDCODED_MOOD:
                 print(f"[💡] Mood detected: (No Face/Neutral - Using {HARDCODED_MOOD})")
                 return HARDCODED_MOOD 
            
            print(f"[💡] Mood detected: {detected_mood.upper()}")
            return detected_mood
        else:
            print(f"Mood API error status {response.status_code}. Using {HARDCODED_MOOD}.")
            return HARDCODED_MOOD
            
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Could not connect to Mood API at {MOOD_API_URL}. Using {HARDCODED_MOOD}.")
        return HARDCODED_MOOD


def controller_to_llm(user_text, step_description="continue the conversation"):
    current_mood = get_user_mood()
    mood_guidance = MOOD_PROMPT_MAPS.get(current_mood, MOOD_PROMPT_MAPS['neutral'])
    
    rich_prompt = (
        f"--- CONTEXT FOR {step_description.upper()} ---\n"
        f"The user's current facial expression/mood is detected as: **{current_mood.upper()}**.\n"
        f"Guidance: {mood_guidance}\n"
        f"--- USER INPUT ---\n"
        f"The user said: '{user_text}'"
    )
    
    return rich_prompt


def robot_say(text):
    print(f"[🤖 Robot]: {text}")
    furhat.say(text=text, blocking=True)

def get_user_input():    
    if USE_KEYBOARD:
        response = input("[⌨️ Type your answer]: ").strip()
        return response
    else:
        print("[👂 Listening...]")
        result = furhat.listen()
        if not result.message:
            robot_say("I didn't quite catch that.")
            return ""
        print(f"[👤 User (Voice)]: {result.message}")
        return result.message

def robot_gesture(name):
    try:
        furhat.gesture(name=name, blocking=False)
    except Exception:
        print(f"Gesture {name} not found.")

# --- MAIN FLOW ---

def run_interview():
    print(f"=== Starting Session (Input Mode: {'KEYBOARD' if USE_KEYBOARD else 'VOICE'}) ===")
    
    # 1. HARDCODED INTRO
    robot_gesture("Smile")
    robot_say("Hello there! I am your study assistant. I'm ready to help you with any academic questions you have.")
    
    # Ask Name
    robot_say("First off, what is your name?")
    user_name = get_user_input()
    
    if not user_name:
        user_name = "Student" 
    
    # Ask Status
    robot_say(f"Nice to meet you, {user_name}. What subject or topic are you working on today?")
    user_topic = get_user_input()

    # 2. SEEDING THE AI (Initial Prompt)
    initial_prompt = controller_to_llm(
        user_text=f"My name is {user_name} and I am working on {user_topic}.",
        step_description="start the interview"
    )
    
    print("[🧠] Thinking...")
    response = chat_session.send_message(initial_prompt)
    ai_text = response.text

    # 3. INTERVIEW LOOP
    for i in range(QUESTIONS_TO_ASK):
        print(f"\n--- Question {i+1} of {QUESTIONS_TO_ASK} ---")
        
        # Robot speaks AI response
        robot_gesture("Nod")
        robot_say(ai_text)
        
        # Get input
        user_answer = get_user_input()
        
        # Logic to handle empty input and prepare the controller prompt
        if not user_answer:
            rich_next_prompt = controller_to_llm(
                user_text="The user didn't respond to the last question/explanation.",
                step_description="get user attention/clarify"
            )
        else:
            rich_next_prompt = controller_to_llm(
                user_text=user_answer,
                step_description="continue the conversation"
            )

        # Send to Gemini (unless it's the last turn)
        if i < QUESTIONS_TO_ASK - 1:
            print("[🧠] Thinking...")
            response = chat_session.send_message(rich_next_prompt)
            ai_text = response.text

    # 4. WRAP UP & SUMMARY
    print("\n=== Wrapping Up ===")
    
    summary_prompt = controller_to_llm(
        user_text="The interview is now over.",
        step_description="summarize and compliment the user's progress"
    )
    
    print("[🧠] Generating Summary...")
    final_response = chat_session.send_message(summary_prompt)
    
    robot_gesture("BigSmile")
    robot_say(final_response.text)

if __name__ == "__main__":
    try:
        run_interview()
    except Exception as e:
        print(f"An error occurred: {e}")