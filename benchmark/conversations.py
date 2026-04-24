"""
10 multi-turn conversation scripts for benchmark.
Each conversation has a theme to test specific memory types.
"""

CONVERSATIONS = [
    {
        "id": 1,
        "theme": "user_preference",
        "description": "User shares preferences, later asks for recommendations",
        "turns": [
            "Hi, my name is Alex and I love Italian food, especially pasta.",
            "I also really enjoy hiking on weekends.",
            "What kind of restaurants would you recommend for me?",
            "What outdoor activities might I enjoy based on what you know about me?",
            "Can you suggest a meal I could cook at home this weekend?",
        ],
    },
    {
        "id": 2,
        "theme": "factual_recall",
        "description": "User provides facts about themselves, agent must recall them",
        "turns": [
            "I work as a software engineer at a startup in Hanoi.",
            "I have 3 years of experience with Python.",
            "My team uses React for the frontend.",
            "What programming languages do I know?",
            "What city do I work in?",
        ],
    },
    {
        "id": 3,
        "theme": "experience_recall",
        "description": "User describes past events, later asks about them",
        "turns": [
            "Last month I attended a machine learning conference in Singapore.",
            "The conference had a great talk about transformer architectures.",
            "I met a researcher from Google Brain there.",
            "What event did I attend last month?",
            "Who did I meet at that event?",
        ],
    },
    {
        "id": 4,
        "theme": "semantic_similarity",
        "description": "User asks conceptually similar questions across turns",
        "turns": [
            "Can you explain how neural networks learn?",
            "That makes sense. How does backpropagation work specifically?",
            "What role does the learning rate play in training?",
            "How is gradient descent related to what we just discussed?",
            "What was the main topic we've been exploring?",
        ],
    },
    {
        "id": 5,
        "theme": "mixed_memory",
        "description": "Mix of preference, facts, and experience recall",
        "turns": [
            "My name is Maria and I prefer Python over JavaScript.",
            "I recently completed a project on sentiment analysis.",
            "The project used BERT and achieved 92% accuracy.",
            "What programming language do I prefer?",
            "Tell me about the project I worked on recently.",
        ],
    },
    {
        "id": 6,
        "theme": "context_continuity",
        "description": "Tests whether agent maintains context across many turns",
        "turns": [
            "Let's plan a trip to Japan.",
            "I want to visit Tokyo and Kyoto.",
            "I have a budget of $3000 for two weeks.",
            "I prefer cultural experiences over shopping.",
            "Summarize the trip plan based on everything I've told you.",
        ],
    },
    {
        "id": 7,
        "theme": "preference_update",
        "description": "User updates a previously stated preference",
        "turns": [
            "I usually prefer tea over coffee.",
            "Actually, I've started drinking coffee lately.",
            "I like it black with no sugar.",
            "What is my current coffee preference?",
            "What drink preference did I mention first?",
        ],
    },
    {
        "id": 8,
        "theme": "technical_qa",
        "description": "Technical Q&A requiring context from earlier turns",
        "turns": [
            "I'm building a RAG system with LangChain.",
            "I'm using Chroma as the vector store.",
            "The documents are research papers in PDF format.",
            "What vector store am I using in my project?",
            "What type of documents am I indexing?",
        ],
    },
    {
        "id": 9,
        "theme": "personal_assistant",
        "description": "Personal assistant scenario with multiple preferences",
        "turns": [
            "I wake up at 6am every day and go for a run.",
            "I'm vegetarian and allergic to nuts.",
            "I work from home and take breaks every 2 hours.",
            "Suggest a morning routine for me.",
            "What dietary restrictions do I have?",
        ],
    },
    {
        "id": 10,
        "theme": "long_term_recall",
        "description": "Tests recall after many unrelated turns",
        "turns": [
            "My favorite programming language is Rust.",
            "Let's talk about something else - what's the capital of France?",
            "Tell me a fun fact about dolphins.",
            "What's 2 + 2?",
            "What's my favorite programming language?",
        ],
    },
    {
        "id": 11,
        "theme": "conflict_update",
        "description": "User corrects a previously stated fact — conflict resolution test",
        "turns": [
            "Tôi dị ứng sữa bò.",
            "Tôi thích màu xanh lá.",
            "À nhầm, tôi dị ứng đậu nành chứ không phải sữa bò.",
            "Tôi có dị ứng gì vậy?",
            "Màu sắc yêu thích của tôi là gì?",
        ],
    },
    {
        "id": 12,
        "theme": "trim_budget",
        "description": "Long multi-turn conversation to trigger context trimming and test token budget management",
        "turns": [
            "My name is David and I'm a data scientist.",
            "I work with Python, Pandas, and Scikit-learn daily.",
            "My current project involves time series forecasting.",
            "I use LSTM and Transformer models for the forecasting task.",
            "The dataset has 5 years of hourly energy consumption data.",
            "We pre-process with z-score normalization and sliding window.",
            "The model achieves MAE of 12.3 on the test set.",
            "We deploy the model via FastAPI on a Kubernetes cluster.",
            "The team has 4 members: 2 engineers and 2 analysts.",
            "What is my name and what do I do professionally?",
        ],
    },
]
