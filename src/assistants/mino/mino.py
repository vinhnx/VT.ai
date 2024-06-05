import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.beta import Assistant
from utils import constants

# Assistant (beta)
# ref: https://platform.openai.com/docs/assistants/tools/code-interpreter/how-it-works


load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=api_key)

NAME = "Mino"

INSTRUCTIONS = """You are a personal math tutor. Write and run code to answer math questions.
Enclose math expressions in $$ (this is helpful to display latex). Example:
```
Given a formula below $$ s = ut + \frac{1}{2}at^{2} $$ Calculate the value of $s$ when $u = 10\frac{m}{s}$ and $a = 2\frac{m}{s^{2}}$ at $t = 1s$
```
"""
# INSTRUCTIONS = "You are an data science expert who is great at analyzing and visualising datasets. take a look at this data"
# "You are a personal math tutor. Write and run code to answer math questions. Your name is Mino."
MODEL = "gpt-4o"


class MinoAssistant:
    def __init__(
        self,
        openai_client: AsyncOpenAI | None,
    ):
        if openai_client is None:
            self.__openai_client__ = AsyncOpenAI(max_retries=2)
        else:
            self.__openai_client__ = openai_client

        self.name = constants.APP_NAME
        self.instructions = INSTRUCTIONS

    async def run_assistant(self) -> Assistant:
        tool = constants.ASSISTANT_TOOL_CODE_INTEPRETER
        assistant_name = NAME
        assistant = await self.__openai_client__.beta.assistants.create(
            name=assistant_name,
            instructions=INSTRUCTIONS,
            tools=[
                {
                    "type": tool,
                }
            ],
            model=MODEL,
        )

        return assistant
