import os
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage
except ImportError:
    ChatGoogleGenerativeAI = None
    HumanMessage = None

class GeminiClient:
    def __init__(self, api_key: str = None, model_name: str = "gemini-pro"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", model_name)
        if not self.api_key:
            print("Warning: GOOGLE_API_KEY not found.")

    def ask_chat(self, prompt: str) -> str:
        """
        Send text prompt to Google Gemini.
        """
        if not ChatGoogleGenerativeAI or not self.api_key:
            return "[Error] API setup missing."
            
        try:
            llm = ChatGoogleGenerativeAI(model=self.model_name, google_api_key=self.api_key)
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"[LLM-failed] {str(e)}"

    def analyze_image(self, image_path: str, prompt: str = "Analyze this chart. What are the key takeaways?") -> str:
        """
        Send image + text to Gemini Vision.
        """
        if not ChatGoogleGenerativeAI or not self.api_key:
            return "[Error] API setup missing."

        if not os.path.exists(image_path):
            return "[Error] Image path not found."

        try:
            # force flash or pro-vision for images
            vision_model = "gemini-1.5-flash" 
            llm = ChatGoogleGenerativeAI(model=vision_model, google_api_key=self.api_key)
            
            import base64
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")

            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"},
                    },
                ]
            )
            
            response = llm.invoke([message])
            return response.content
            
        except Exception as e:
            return f"[Vision-failed] {str(e)}"
