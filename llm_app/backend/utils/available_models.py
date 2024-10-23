"""
This module contains the available models for the LLM app.

Attributes
----------

OPENAIMODELS: dict
    A dictionary containing the available OpenAI models.
ANTHROPICMODELS: dict
    A dictionary containing the available Anthropic models.
"""

OPENAIMODELS = {
    "GPT-4o": {
        "api": "chatgpt-4o-latest",
        "max_output": 16384,
        "context_window": 128000,
    },
    "GPT-4o Mini": {
        "api": "gpt-4o-mini-2024-07-18",
        "max_output": 16000,
    },
    "GPT-4 Turbo": {
        "api": "gpt-4-turbo-preview",
        "max_output": 4096,
    },
    "GPT-4": {
        "api": "gpt-4",
        "max_output": 8192,
    },
    # As of July 2024, gpt-4o-mini should be used in place of gpt-3.5-turbo,
    # as it is cheaper, more capable, multimodal, and just as fast. gpt-3.5-turbo
    # is still available for use in the API.
    "GPT-3.5 Turbo": {
        "api": "gpt-3.5-turbo",
        "max_output": 4096,
    },
    "GPT-3.5 Turbo 16k": {
        "api": "gpt-3.5-turbo-16k",
        "max_output": 16384,
    },
}


ANTHROPICMODELS = {
    "Claude 3.5 Sonnet": {
        "api": "claude-3-5-sonnet-latest",
        "max_output": 8192,
    },
    "Claude 3 Opus": {
        "api": "claude-3-opus-20240229",
        "max_output": 4096,
    },
    "Claude 3 Haiku": {
        "api": "claude-3-haiku-20240307",
        "max_output": 4096,
    },
}


#  { 10/10/24
#       "id": "tts-1",
#       "object": "model",
#       "created": 1681940951,
#       "owned_by": "openai-internal"
#     },
#     {
#       "id": "tts-1-1106",
#       "object": "model",
#       "created": 1699053241,
#       "owned_by": "system"
#     },
#     {
#       "id": "chatgpt-4o-latest",
#       "object": "model",
#       "created": 1723515131,
#       "owned_by": "system"
#     },
#     {
#       "id": "dall-e-2",
#       "object": "model",
#       "created": 1698798177,
#       "owned_by": "system"
#     },
#     {
#       "id": "gpt-4o-2024-08-06",
#       "object": "model",
#       "created": 1722814719,
#       "owned_by": "system"
#     },
#     {
#       "id": "gpt-4-turbo-preview",
#       "object": "model",
#       "created": 1706037777,
#       "owned_by": "system"
#     },
#     {
#       "id": "gpt-4o",
#       "object": "model",
#       "created": 1715367049,
#       "owned_by": "system"
#     },
#     {
#       "id": "gpt-3.5-turbo-instruct",
#       "object": "model",
#       "created": 1692901427,
#       "owned_by": "system"
#     },
#     {
#       "id": "gpt-4-0125-preview",
#       "object": "model",
#       "created": 1706037612,
#       "owned_by": "system"
#     },
#     {
#       "id": "gpt-3.5-turbo-0125",
#       "object": "model",
#       "created": 1706048358,
#       "owned_by": "system"
#     },
#     {
#       "id": "gpt-3.5-turbo",
#       "object": "model",
#       "created": 1677610602,
#       "owned_by": "openai"
#     },
#     {
#       "id": "babbage-002",
#       "object": "model",
#       "created": 1692634615,
#       "owned_by": "system"
#     },
#     {
#       "id": "davinci-002",
#       "object": "model",
#       "created": 1692634301,
#       "owned_by": "system"
#     },
#     {
#       "id": "dall-e-3",
#       "object": "model",
#       "created": 1698785189,
#       "owned_by": "system"
#     },
#     {
#       "id": "gpt-4-turbo-2024-04-09",
#       "object": "model",
#       "created": 1712601677,
#       "owned_by": "system"
#     },
#     {
#       "id": "tts-1-hd",
#       "object": "model",
#       "created": 1699046015,
#       "owned_by": "system"
#     },
#     {
#       "id": "tts-1-hd-1106",
#       "object": "model",
#       "created": 1699053533,
#       "owned_by": "system"
#     },
#     {
#       "id": "gpt-4-1106-preview",
#       "object": "model",
#       "created": 1698957206,
#       "owned_by": "system"
#     },
#     {
#       "id": "text-embedding-ada-002",
#       "object": "model",
#       "created": 1671217299,
#       "owned_by": "openai-internal"
#     },
#     {
#       "id": "gpt-3.5-turbo-16k",
#       "object": "model",
#       "created": 1683758102,
#       "owned_by": "openai-internal"
#     },
#     {
#       "id": "gpt-4o-realtime-preview-2024-10-01",
#       "object": "model",
#       "created": 1727131766,
#       "owned_by": "system"
#     },
#     {
#       "id": "text-embedding-3-small",
#       "object": "model",
#       "created": 1705948997,
#       "owned_by": "system"
#     },
#     {
#       "id": "whisper-1",
#       "object": "model",
#       "created": 1677532384,
#       "owned_by": "openai-internal"
#     },
#     {
#       "id": "text-embedding-3-large",
#       "object": "model",
#       "created": 1705953180,
#       "owned_by": "system"
#     },
#     {
#       "id": "gpt-4-turbo",
#       "object": "model",
#       "created": 1712361441,
#       "owned_by": "system"
#     },
#     {
#       "id": "gpt-4o-2024-05-13",
#       "object": "model",
#       "created": 1715368132,
#       "owned_by": "system"
#     },
#     {
#       "id": "gpt-3.5-turbo-1106",
#       "object": "model",
#       "created": 1698959748,
#       "owned_by": "system"
#     },
#     {
#       "id": "gpt-4-0613",
#       "object": "model",
#       "created": 1686588896,
#       "owned_by": "openai"
#     },
#     {
#       "id": "gpt-4",
#       "object": "model",
#       "created": 1687882411,
#       "owned_by": "openai"
#     },
#     {
#       "id": "gpt-3.5-turbo-instruct-0914",
#       "object": "model",
#       "created": 1694122472,
#       "owned_by": "system"
#     },
#     {
#       "id": "gpt-4o-mini",
#       "object": "model",
#       "created": 1721172741,
#       "owned_by": "system"
#     },
#     {
#       "id": "gpt-4o-realtime-preview",
#       "object": "model",
#       "created": 1727659998,
#       "owned_by": "system"
#     },
#     {
#       "id": "gpt-4o-mini-2024-07-18",
#       "object": "model",
#       "created": 1721172717,
#       "owned_by": "system"
#     }
