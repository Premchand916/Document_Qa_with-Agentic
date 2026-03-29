from dataclasses import dataclass, field


@dataclass
class SimpleConversationMemory:
    """Small drop-in replacement for ConversationBufferMemory.

    The current app stores chat history directly in Streamlit state, but this
    helper keeps the old import path usable for modules or experiments that
    still expect a memory object.
    """

    memory_key: str = "chat_history"
    return_messages: bool = True
    messages: list = field(default_factory=list)

    def load_memory_variables(self, _inputs=None):
        if self.return_messages:
            return {self.memory_key: list(self.messages)}

        transcript = "\n".join(
            f"{message.get('role', 'user')}: {message.get('content', '')}"
            for message in self.messages
        )
        return {self.memory_key: transcript}

    def save_context(self, inputs, outputs):
        user_text = ""
        assistant_text = ""

        if isinstance(inputs, dict):
            user_text = next((str(value) for value in inputs.values() if value), "")
        if isinstance(outputs, dict):
            assistant_text = next((str(value) for value in outputs.values() if value), "")

        if user_text:
            self.add_user_message(user_text)
        if assistant_text:
            self.add_ai_message(assistant_text)

    def add_user_message(self, content):
        self.messages.append({"role": "user", "content": str(content)})

    def add_ai_message(self, content):
        self.messages.append({"role": "assistant", "content": str(content)})

    def clear(self):
        self.messages.clear()


def get_memory():
    return SimpleConversationMemory(
        memory_key="chat_history",
        return_messages=True,
    )
