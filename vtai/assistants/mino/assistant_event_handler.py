from openai import AssistantEventHandler
from typing_extensions import override


# ref: https://platform.openai.com/docs/assistants/overview
class EventHandler(AssistantEventHandler):
    @override
    def on_text_created(self, text) -> None:
        print("\nassistant > ", end="", flush=True)

    @override
    def on_text_delta(self, delta, snapshot):
        print(delta.value, end="", flush=True)

    def on_tool_call_created(self, tool_call):
        print(f"\nassistant > {tool_call.type}\n", flush=True)

    def on_tool_call_delta(self, delta, snapshot):
        if delta.type == "code_interpreter":
            if delta.code_interpreter.input:
                print(delta.code_interpreter.input, end="", flush=True)
            if delta.code_interpreter.outputs:
                print("\n\noutput >", flush=True)
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        print(f"\n{output.logs}", flush=True)


# Then, we use the `stream` SDK helper
# with the `EventHandler` class to create the Run
# and stream the response.
# with client.beta.threads.runs.stream(
#     thread_id=thread.id,
#     assistant_id=assistant.id,
#     instructions="Please address the user as Jane Doe. The user has a premium account.",
#     event_handler=EventHandler(),
# ) as stream:
#     stream.until_done()
