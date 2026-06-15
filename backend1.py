
from deepagents.backends import FilesystemBackend
from utils import get_model
from deepagents import create_deep_agent


def get_article_points(topic: str) -> str:
    """Simple tool that gives article talking points

    Args:
        topic (str): topic

    Returns:
        str: Talking points
    """
    return f"""
Article Topic: {topic}

Talking Points:

1. Introduce what an economy is
2. Explain why economies matter to everyday people
3. Discuss India's economic journey since independence
4. Explain the 1991 economic reforms
5. Highlight key growth drivers:
   - Agriculture
   - Industry
   - Services
   - Digital Transformation
6. Give a practical real-world example
7. Discuss current challenges
8. End with future opportunities and a summary
"""


model = get_model()

backend = FilesystemBackend(
    root_dir="./Agent_workspace",
    virtual_mode=True
)

agent = create_deep_agent(
    model=model,
    tools=[get_article_points],
    backend=backend,
    system_prompt="""
You are a technical article writing agent.

You MUST complete ALL steps.

Do not stop after creating todos.md.

Required workflow:

1. Create /todos.md
2. Call get_article_points tool
3. Create /notes.md
4. Create /article.md
5. Update todos.md so every task becomes [completed]
6. Return summary

The task is NOT complete until all three files exist:
- /todos.md
- /notes.md
- /article.md
""")

if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Write an article on India's Economy: A Story of Growth and Opportunity."
                }
            ]
        }
    )

    result["messages"][-1].pretty_print()

