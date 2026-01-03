"""Reason-and-Act Knowledge Retrieval Agent via the OpenAI Agent SDK."""

import asyncio
import contextlib
import logging
import signal
import sys

import agents
import gradio as gr
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from agents import OpenAIChatCompletionsModel


from src.prompts import REACT_INSTRUCTIONS
from src.utils import (
    AsyncWeaviateKnowledgeBase,
    Configs,
    get_weaviate_async_client,
    oai_agent_stream_to_gradio_messages,
)

load_dotenv(verbose=True)


logging.basicConfig(level=logging.INFO)

class EmailContent(BaseModel):
    subject: str = Field(
        description="The subject line of the email. Should be concise and descriptive."
    )
    body: str = Field(
        description="The main content of the email. Should be well-formatted with proper greeting, paragraphs, and signature."
    )
AGENT_LLM_NAME = "gemini-2.5-flash"


@agents.function_tool
def submit_email(subject: str, body: str) -> dict:
    """Submit the final generated email."""
    return {
        "subject": subject,
        "body": body,
    }


REACT_INSTRUCTIONS = """
        You are an Email Generation Assistant.
        Your task is to generate a professional email based on the user's request.

        GUIDELINES:
        - Create an appropriate subject line (concise and relevant)
        - Write a well-structured email body with:
            * Professional greeting
            * Clear and concise main content
            * Appropriate closing
            * Your name as signature
        - Suggest relevant attachments if applicable (empty list if none needed)
        - Email tone should match the purpose (formal for business, friendly for colleagues)
        - Keep emails concise but complete

        IMPORTANT RULES:
        - You MUST call the tool `submit_email` with the final result.
        - Do NOT output text directly.
        - Do NOT explain your reasoning.
    """


configs = Configs.from_env_var()
async_weaviate_client = get_weaviate_async_client(
    http_host=configs.weaviate_http_host,
    http_port=configs.weaviate_http_port,
    http_secure=configs.weaviate_http_secure,
    grpc_host=configs.weaviate_grpc_host,
    grpc_port=configs.weaviate_grpc_port,
    grpc_secure=configs.weaviate_grpc_secure,
    api_key=configs.weaviate_api_key,
)
async_openai_client = AsyncOpenAI()
async_knowledgebase = AsyncWeaviateKnowledgeBase(
    async_weaviate_client,
    collection_name="enwiki_20250520",
)


async def _cleanup_clients() -> None:
    """Close async clients."""
    await async_weaviate_client.close()
    await async_openai_client.close()


def _handle_sigint(signum: int, frame: object) -> None:
    """Handle SIGINT signal to gracefully shutdown."""
    with contextlib.suppress(Exception):
        asyncio.get_event_loop().run_until_complete(_cleanup_clients())
    sys.exit(0)

# model = OpenAIChatCompletionsModel(
#     model=AGENT_LLM_NAME,
#     openai_client=async_openai_client,
#     response_format=EmailContent,   # <-- schema enforcement
# )



async def _main(question: str, gr_messages: list[ChatMessage]):
    main_agent = agents.Agent(
        name="E-mail Agent",
        instructions=REACT_INSTRUCTIONS,
        tools=[submit_email],
        model=OpenAIChatCompletionsModel(
            model=AGENT_LLM_NAME,
            openai_client=async_openai_client,
        ),
    )

    result_stream = agents.Runner.run_streamed(main_agent, input=question)
    async for _item in result_stream.stream_events():
        gr_messages += oai_agent_stream_to_gradio_messages(_item)
        if len(gr_messages) > 0:
            yield gr_messages


demo = gr.ChatInterface(
    _main,
    title="2.1 OAI Agent SDK ReAct",
    type="messages",
    examples=[
        "Write an email requesting the team to run the pipeline before month end.",
        "Write an email to cancel my AGO subscription.",
    ],
)


if __name__ == "__main__":
    configs = Configs.from_env_var()


    async_openai_client = AsyncOpenAI()
    agents.set_tracing_disabled(disabled=True)

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        demo.launch(share=True)
    finally:
        asyncio.run(_cleanup_clients())
