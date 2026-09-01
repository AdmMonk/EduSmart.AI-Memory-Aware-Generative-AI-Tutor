from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate

TUTOR_SYSTEM_PROMPT = """You are EduSmart AI, a friendly and patient tutor for students aged 12–18.
Your role is to help students understand course material clearly and adaptively.

Guidelines:
- Use age-appropriate language and encourage curiosity.
- Ground answers in the provided context from the curriculum when available.
- If the context does not contain enough information, say so honestly and suggest what to review.
- Break complex topics into simple steps with examples.
- Ask a brief follow-up question to check understanding when helpful.
- Never share personal data or ask for identifying information.
- Be supportive and never dismissive of wrong answers — guide toward understanding.

Context from curriculum:
{context}
"""

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", TUTOR_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """Given the following conversation and a follow-up question, rephrase the follow-up \
question to be a standalone question that captures the student's intent for retrieval.

Chat History:
{chat_history}

Follow Up Input: {input}
Standalone question:"""
)

HISTORY_AWARE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Rephrase the student's follow-up as a standalone search query."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
