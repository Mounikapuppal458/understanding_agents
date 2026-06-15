from deepagents.backends import FilesystemBackend, CompositeBackend, StateBackend
from utils import get_model
from deepagents import create_deep_agent


def get_article_points(topic: str) -> str:
    """Simple tool that gives article talking points"""

    return f"""
Article Topic: {topic}

Talking Points:
1. Introduction to the Human Body
2. Explain the basic organization of the body
3. Discuss major body systems:
   - Skeletal System
   - Muscular System
   - Nervous System
   - Circulatory System
   - Respiratory System
   - Digestive System
4. Explain how these systems work together
5. Give real-life examples of body functions
6. Discuss the importance of maintaining body health
7. End with a summary and future understanding of human anatomy
"""


model = get_model()

backend = CompositeBackend(
    default=StateBackend(),
    routes={
        "/workspace/": FilesystemBackend(
            root_dir="./agent_workspace",
            virtual_mode=True
        ),
    },
)

agent = create_deep_agent(
    model=model,
    tools=[get_article_points],
    backend=backend,
    system_prompt="""
You are an educational article writing agent.

Workflow:

1. Create a short plan and write all todos to /workspace/todos.md
2. Use the get_article_points tool
3. Write rough notes to /workspace/roughnotes.md
4. Write the final article to /workspace/article.md
5. Update todos.md to mark all tasks as completed
6. Return a summary mentioning all created files

Topic:
Human Body and Its Architecture

Requirements:
- Beginner friendly
- Simple language
- Use practical examples
- Explain each body system clearly
- Include a conclusion
"""
)

if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Write an article on Human Body and Its Architecture."
                }
            ]
        }
    )

    result["messages"][-1].pretty_print()

