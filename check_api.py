import os
import sys
import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

def check_api_key():
    """
    Standalone script to check if the GOOGLE_API_KEY is present and valid.
    Specifically tests the key against Google's Gemini API.
    """
    # Load .env file
    dotenv.load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found in .env file or environment.")
        print("Please add 'GOOGLE_API_KEY=your_key_here' to your .env file.")
        return False
    
    print(f"🔍 API Key found: {api_key[:5]}...{api_key[-5:]}")
    print("⏳ Testing API connection with Gemini...")
    
    try:
        # Attempt a simple model call
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key
        )
        
        # We use a very simple query to minimize tokens
        response = llm.invoke("Hello, are you there?")
        
        if response:
            print("✅ Success! The API key is VALID and working correctly.")
            print(f"🤖 Response received: {response.content[:50]}...")
            return True
        else:
            print("❌ Error: Received empty response from the API.")
            return False
            
    except Exception as e:
        print("❌ Error: The API key appears to be INVALID or there is a connection issue.")
        print(f"⚠️ Details: {str(e)}")
        
        if "API_KEY_INVALID" in str(e):
            print("\n💡 Tip: Double-check your API key at https://aistudio.google.com/app/apikey")
        elif "quota" in str(e).lower():
            print("\n💡 Tip: You might have exceeded your API quota.")
            
        return False

if __name__ == "__main__":
    success = check_api_key()
    sys.exit(0 if success else 1)
