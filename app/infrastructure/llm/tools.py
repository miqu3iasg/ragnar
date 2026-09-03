# Ref:
# - OpenAI function calling format (OpenRouter follows the same schema): #https://platform.openai.com/docs/guides/function-calling
#
# This module only defines *what* tools are available to the model (their
# JSON schema). It does not execute them; dispatch happens in
# domain/research/service.py, which owns the loop that decides what to do
# when the model requests a tool call (run the full retrieval pipeline,
# then call the model again with the retrieved context).

SEARCH_WEB_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": (
            "Search the web for up-to-date information relevant to the "
            "user's question. Use this when answering requires facts, "
            "recent events, or sources beyond your own knowledge — not for "
            "questions you can already answer confidently on your own."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, in the same language as the user's question.",
                }
            },
            "required": ["query"],
        },
    },
}

AVAILABLE_TOOLS = [SEARCH_WEB_TOOL]
