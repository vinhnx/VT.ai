from openai import OpenAI
from utils import constants

# Assistant (beta)
# ref: https://platform.openai.com/docs/assistants/tools/code-interpreter/how-it-works

NAME = "Mino"
INSTRUCTIONS = "You are a personal math tutor. Write and run code to answer math questions. Your name is Mino."
FUNCTION_NAME = "code_interpreter"
MODEL = "gpt-4-turbo"


class MinoAssistant:
    def __init__(
        self,
        openai_client=OpenAI(max_retries=2),
    ):
        if openai_client is None:
            self.__openai_client__ = OpenAI(max_retries=2)
            # self.__openai_client__ = AsyncOpenAI(max_retries=2)
        else:
            self.__openai_client__ = openai_client

        self.name = constants.APP_NAME
        self.instructions = INSTRUCTIONS
        self.assistant = self.__openai_client__.beta.assistants.create(
            name=self.name,
            instructions=self.instructions,
            tools=[
                {
                    "type": FUNCTION_NAME,
                }
            ],
            model=MODEL,
        )

        self.thread = self.__openai_client__.beta.threads.create()

    def run_assistant(self, query: str) -> str:
        run = self.__openai_client__.beta.threads.runs.create_and_poll(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
            instructions=query,
        )

        if run.status == "completed":
            response = self.__openai_client__.beta.threads.messages.list(
                thread_id=self.thread.id
            )

            response_message = response.data[0].content[0].text.value
            print(response_message)
            return response_message
        else:
            return "unknown"
