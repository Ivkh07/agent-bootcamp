"""Code Interpreter example.

Logs traces to LangFuse for observability and evaluation.

You will need your E2B API Key.
"""

from pathlib import Path
import json
import agents
import gradio as gr
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from openai import AsyncOpenAI
import os

from src.utils import (
    CodeInterpreter,
    oai_agent_stream_to_gradio_messages,
    pretty_print,
    set_up_logging,
    setup_langfuse_tracer,
)
from src.utils.langfuse.shared_client import langfuse_client


load_dotenv(verbose=True)

set_up_logging()

CODE_INTERPRETER_INSTRUCTIONS = """\
The `code_interpreter` tool executes Python commands. \
Please note that data is not persisted. Each time you invoke this tool, \
you will need to run import and define all variables from scratch.

You can access the local filesystem using this tool. \
Instead of asking the user for file inputs, you should try to find the file \
using this tool.

Recommended packages: Pandas, Numpy, SymPy, Scikit-learn.

You can also run Jupyter-style shell commands (e.g., `!pip freeze`)
but you won't be able to install packages.
to answer questions use file named 'data_b.csv'

"""

AGENT_LLM_NAME = "gemini-2.5-flash"#"gemini-2.5-"pro
async_openai_client = AsyncOpenAI()
code_interpreter = CodeInterpreter(
    local_files=[
        Path("sandbox_content/"),
        "tests/tool_tests/example_files/data_b.csv",
        #Path  ('/home/coder/data/ready'), 
        # ("/home/coder/data/1"),  
        #("tests/tool_tests/example_files/example_a.csv"),
    ], 
    sandbox_output_file = 'result.json', 
    local_save_directory = "local_sandbox_downloads"
)


async def _main(question: str, gr_messages: list[ChatMessage]):
    setup_langfuse_tracer()

    main_agent = agents.Agent(
        name="Data Analysis Agent",
        instructions=CODE_INTERPRETER_INSTRUCTIONS,
        tools=[
            agents.function_tool(
                code_interpreter.run_code,
                name_override="code_interpreter",
            )
        ],
        model=agents.OpenAIChatCompletionsModel(
            model=AGENT_LLM_NAME, openai_client=async_openai_client
        ),
    )

    with langfuse_client.start_as_current_span(name="ik_Agents-SDK-Trace") as span:
        span.update(input=question)

        result_stream = agents.Runner.run_streamed(main_agent, input=question)
        async for _item in result_stream.stream_events():
            gr_messages += oai_agent_stream_to_gradio_messages(_item)
            if len(gr_messages) > 0:
                yield gr_messages

        span.update(output=result_stream.final_output)

    pretty_print(gr_messages)
    yield gr_messages


demo = gr.ChatInterface(
    _main,
    title="2.1 OAI Agent SDK ReAct + LangFuse Code Interpreter",
    type="messages",
    examples=[
        "How many columns are in the file data_b.csv? Save answer n a file named 'result.csv' ",
        "How many columns are in the file data_b.csv? Save answer as a dictionary in file named 'result.csv' ",
        "Build a clustering model using data_b.csv. Drop the columns id, date, client_id, card_id, card_number, expires, and cvv. Use only numerical features; treat mcc as categorical and encode it into 5 features. Impute missing numerical values with the median. Compute the silhouette score and provide descriptive insights for each cluster. Save the output in result.csv.",
    ],
)


if __name__ == "__main__":
    demo.launch(share=True)
