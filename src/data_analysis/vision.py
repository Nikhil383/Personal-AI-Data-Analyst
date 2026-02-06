import os
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage
except ImportError:
    ChatGoogleGenerativeAI = None
    HumanMessage = None

def analyze_plot(image_path: str, api_key: str = None) -> str:
    """
    Reflect on a generated plot using a Vision model.
    """
    if not ChatGoogleGenerativeAI:
        return "[Vision-error] langchain-google-genai not installed."
        
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        return "[Vision-error] No API Key provided."

    if not os.path.exists(image_path):
        return "[Vision-error] Image path not found."

    try:
        # gemini-1.5-flash is good for vision and speed
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
        
        # Load image data
        import base64
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        prompt = "Analyze this chart. What are the key takeaways or trends visible? Be concise (2-3 sentences)."
        
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
