"""
Starter prompt and command utilities for VT.ai application.

This module provides a collection of high-quality hardcoded starter prompts, commands,
and functions to manage them. It includes the ability to generate route-based
starter prompts and commands as alternative approaches.
"""

import random
from random import choice, shuffle
from typing import Dict, List, Optional, Union

import chainlit as cl
from router.constants import SemanticRouterType
from router.trainer import create_routes

# Comprehensive set of hardcoded starter prompts covering various use cases
STARTER_PROMPTS = [
    # Code assistance prompts
    {
        "label": "Help with Python Code",
        "message": "I'm trying to write a function that finds all prime numbers up to a given limit using the Sieve of Eratosthenes algorithm. Can you help me implement this efficiently in Python?",
    },
    {
        "label": "Code Review Request",
        "message": "Could you review this function that calculates Fibonacci numbers? I'm concerned about its performance with large inputs:\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```",
    },
    {
        "label": "API Integration",
        "message": "I need to integrate a REST API into my application. Can you show me how to make API requests with proper error handling and retry logic using Python's requests library?",
    },
    # Data analysis prompts
    {
        "label": "Data Visualization Help",
        "message": "I have a dataset of customer purchases over time. What are some effective visualization approaches I could use to identify trends and patterns in this data? Please provide examples with matplotlib or seaborn.",
    },
    {
        "label": "SQL Query Optimization",
        "message": "I have a slow-performing SQL query that joins multiple tables and uses several WHERE conditions. Can you explain some strategies for optimizing it and making it more efficient?",
    },
    {
        "label": "ML Model Selection",
        "message": "I'm working on a project to predict housing prices based on features like location, size, number of rooms, etc. Which machine learning algorithms would be most appropriate for this task and why?",
    },
    # Creative writing prompts
    {
        "label": "Blog Post Outline",
        "message": "I'm writing a blog post about the environmental impact of AI model training. Could you help me create an engaging outline with key points I should cover?",
    },
    {
        "label": "Story Idea Generator",
        "message": "I'm looking for inspiration for a science fiction short story. Could you generate a creative premise involving artificial intelligence in a post-apocalyptic setting?",
    },
    # Image generation prompts
    {
        "label": "Generate Image",
        "message": "Create an image of a futuristic city with flying vehicles and vertical gardens on skyscrapers. The city should be bathed in soft morning light with mountains in the background.",
    },
    {
        "label": "Logo Design Concept",
        "message": "I need a logo concept for a tech startup called 'NeuralSphere' that focuses on AI-powered healthcare solutions. The logo should be modern, professional, and incorporate neural network imagery.",
    },
    # Learning/education prompts
    {
        "label": "Explain Transformer Models",
        "message": "Could you explain how transformer models work in natural language processing? I understand basic neural networks but I'm trying to grasp attention mechanisms and self-supervision.",
    },
    {
        "label": "Learning Roadmap",
        "message": "I want to learn full-stack web development. Can you create a structured learning path for me, going from beginner to advanced topics? Please include recommended resources for each stage.",
    },
    # Problem-solving prompts
    {
        "label": "Debugging Help",
        "message": "My web application is experiencing memory leaks in production. What debugging approaches and tools would you recommend to identify and fix the root cause?",
    },
    {
        "label": "System Design Challenge",
        "message": "I need to design a scalable notification system that can handle millions of users and support multiple delivery channels (email, SMS, push). What architecture would you recommend?",
    },
    # Planning/organization prompts
    {
        "label": "Project Management Plan",
        "message": "I'm leading a software development project with a team of 5 engineers. Can you help me create a comprehensive project management plan including milestones, task breakdowns, and risk assessment?",
    },
    {
        "label": "Meeting Agenda Template",
        "message": "I need to organize effective sprint planning meetings for my agile development team. Can you provide a structured agenda template with time allocations and key discussion points?",
    },
    # General queries
    {
        "label": "Technology Comparison",
        "message": "What are the key differences between NoSQL and SQL databases? When would you recommend using each type, and what are their respective advantages and limitations?",
    },
    {
        "label": "Future Tech Trends",
        "message": "What emerging technologies do you think will have the biggest impact on software development in the next 5 years, and how should developers prepare for these changes?",
    },
    {
        "label": "Ethical AI Discussion",
        "message": "What are the most important ethical considerations when developing AI systems, and what practices can developers implement to ensure their AI applications are responsible and fair?",
    },
    {
        "label": "Productivity Tips",
        "message": "Can you suggest some evidence-based productivity techniques and tools specifically for software developers to manage complex projects and reduce mental fatigue?",
    },
    # Research and analysis prompts
    {
        "label": "Literature Review",
        "message": "I'm starting a research project on the environmental impact of blockchain technologies. Can you help me synthesize the key findings from recent academic literature on this topic?",
    },
    {
        "label": "Trend Analysis",
        "message": "What are the emerging trends in quantum computing research, and how might these developments affect cybersecurity in the next decade? Please cite specific advancements and their potential implications.",
    },
    {
        "label": "Comparative Analysis",
        "message": "I'd like to compare the approaches to reinforcement learning used in AlphaGo, MuZero, and other game-playing AI systems. What are the key innovations and trade-offs in each approach?",
    },
    # Business and entrepreneurship prompts
    {
        "label": "Startup Idea Validation",
        "message": "I have an idea for a SaaS platform that helps indie game developers manage community feedback. Can you help me validate this concept by identifying potential challenges, competitive advantages, and market opportunities?",
    },
    {
        "label": "Business Model Canvas",
        "message": "I'm launching a subscription service for AI-optimized meal planning. Can you help me create a Business Model Canvas that outlines key partners, activities, resources, value propositions, customer relationships, channels, customer segments, cost structure, and revenue streams?",
    },
    {
        "label": "Marketing Strategy",
        "message": "I've developed a new developer tool that improves code review workflows. What marketing strategies would be most effective for reaching software engineering teams? Please consider content marketing, community building, and other relevant approaches.",
    },
    # AI and machine learning prompts
    {
        "label": "Prompt Engineering Tips",
        "message": "I'm working with large language models and need to improve my prompt engineering skills. Can you share advanced techniques for crafting effective prompts, including examples of how to use few-shot learning and chain-of-thought reasoning?",
    },
    {
        "label": "ML Project Workflow",
        "message": "I'm setting up an ML workflow for a computer vision project. Can you outline best practices for data versioning, experiment tracking, model evaluation, and deployment? Please recommend specific tools where appropriate.",
    },
    {
        "label": "Fine-tuning Strategy",
        "message": "I want to fine-tune a language model for legal document analysis. What approach would you recommend regarding dataset preparation, training strategies, and evaluation metrics? Please also address potential challenges and how to overcome them.",
    },
    # Health and wellness prompts
    {
        "label": "Ergonomic Workspace",
        "message": "As a software developer who spends 8+ hours daily at a computer, I want to optimize my workspace for ergonomics and health. What specific equipment, arrangements, and habits would you recommend based on current research?",
    },
    {
        "label": "Developer Wellness Plan",
        "message": "Can you create a comprehensive wellness plan for technology professionals that addresses physical health, mental well-being, and preventing burnout? I'm looking for practical strategies that can be integrated into a busy work schedule.",
    },
    # Design and UX/UI prompts
    {
        "label": "UX Research Methods",
        "message": "I'm designing a new developer tool and need to conduct user research. What user research methods would be most effective for understanding developer workflows and pain points? Please describe specific approaches and how to analyze the results.",
    },
    {
        "label": "Design System Creation",
        "message": "I need to create a design system for my company's suite of enterprise applications. What components, guidelines, and documentation should I include to ensure consistency and accessibility while enabling rapid development?",
    },
    # Document processing prompts
    {
        "label": "Documentation Generator",
        "message": "I have a Python codebase with minimal documentation. Can you suggest an approach to automatically generate comprehensive documentation from my code? Please include specific tools and best practices for maintaining documentation over time.",
    },
    {
        "label": "Technical Writing Guide",
        "message": "I need to create technical documentation for an open-source project. Can you provide a guide on writing clear, accessible documentation including README, API docs, and tutorials? Please include formats, structure, and examples.",
    },
    # Mathematical problems
    {
        "label": "Algorithm Complexity",
        "message": "I'm trying to understand the time and space complexity of different sorting algorithms. Can you explain the Big O notation for common sorting algorithms (quicksort, mergesort, heapsort, etc.) and when to use each one based on their performance characteristics?",
    },
    {
        "label": "Statistical Analysis Help",
        "message": "I have a dataset of user engagement metrics before and after a UI change. What statistical tests should I use to determine if the changes had a significant impact on user behavior? Please explain the approach and how to interpret the results.",
    },
    # Career development
    {
        "label": "Technical Interview Prep",
        "message": "I have a technical interview for a senior backend developer position next week. Can you help me prepare by suggesting a study plan focusing on system design, algorithms, and behavioral questions? Please include specific practice exercises.",
    },
    {
        "label": "Tech Resume Review",
        "message": "I'd like to optimize my resume for a machine learning engineer position. What skills, projects, and experiences should I highlight? Please provide a template or structure that effectively showcases technical expertise.",
    },
    # DevOps and infrastructure
    {
        "label": "CI/CD Pipeline Design",
        "message": "I'm setting up a CI/CD pipeline for a microservices application. Can you outline the components and tools I should include for efficient testing, building, and deployment? Please consider security scanning, environment management, and monitoring.",
    },
    {
        "label": "Kubernetes Architecture",
        "message": "I need to design a Kubernetes architecture for a web application with variable load. Can you recommend an appropriate cluster structure, resource allocation, scaling policies, and monitoring setup?",
    },
    # Security and privacy
    {
        "label": "Security Code Review",
        "message": "What are the most common security vulnerabilities in web applications, and how can I systematically check for them during code reviews? Please include examples of how these vulnerabilities might appear in code and how to fix them.",
    },
    {
        "label": "Privacy by Design",
        "message": "I'm developing a mobile app that collects user location data. How can I implement privacy by design principles to ensure compliance with regulations like GDPR and CCPA? Please include specific technical approaches and user communication strategies.",
    },
]

# Command definitions with corresponding icons and descriptions
# Each command maps to a specific router intent for semantic routing
COMMANDS = [
    {
        "id": "Code",
        "icon": "code",
        "description": "Get programming assistance",
        "button": True,
        "persistent": False,
        "route": SemanticRouterType.TEXT_PROCESSING,  # Maps to text-processing for now
    },
    {
        "id": "Image",
        "icon": "image",
        "description": "Generate or analyze images",
        "button": True,
        "persistent": False,
        "route": SemanticRouterType.IMAGE_GENERATION,  # Maps to image-generation
    },
    {
        "id": "Search",
        "icon": "search",
        "description": "Search the web for information",
        "button": True,
        "persistent": False,
        "route": SemanticRouterType.WEB_SEARCH,  # Maps to web-search
    },
]

# Command templates - Example messages that will be inserted when a command is selected
COMMAND_TEMPLATES = {
    "Code": [
        "Help me implement a function that {goal} in {language}.",
        "Review this code and suggest improvements: ```{language}\n{code}\n```",
        "Explain how to properly implement {feature} in {language}.",
        "What's the best way to structure a {project_type} project in {language}?",
        "Show me how to create a {component} for my {framework} application.",
    ],
    "Image": [
        "Generate an image of {subject} with {style} style.",
        "Create a photorealistic image of {subject} with high detail.",
        "Generate an artistic rendering of {scene} in the style of {artist}.",
        "Create a logo for {company_name} that represents {values}.",
        "Design a user interface for a {app_type} application with {theme} theme.",
    ],
    "Explain": [
        "Explain {concept} in simple terms.",
        "What is {topic} and how does it work?",
        "Compare {thing1} and {thing2}, highlighting key differences.",
        "I want to learn about {subject}. Where should I start?",
        "What are the main principles behind {field}?",
    ],
    "Data": [
        "Help me analyze this dataset: {data}",
        "What statistical methods should I use to {goal}?",
        "How can I visualize {data_type} data effectively?",
        "Help me interpret these results: {results}",
        "What's the best machine learning approach for {problem}?",
    ],
    "Write": [
        "Help me write a {document_type} about {topic}.",
        "Create an outline for a {content_type} on {subject}.",
        "Suggest improvements for this text: {text}",
        "Write a {tone} email to {recipient} regarding {subject}.",
        "Help me create compelling content for my {platform} about {topic}.",
    ],
    "Plan": [
        "Create a roadmap for {project} with key milestones.",
        "Help me organize a {event_type} with {participants} participants.",
        "What's a good project management approach for {project_type}?",
        "Break down this goal into actionable steps: {goal}",
        "Create a schedule for {activity} over the next {time_period}.",
    ],
    "Debug": [
        "I'm getting this error: {error}. How can I fix it?",
        "My {software} is {problem}. What could be causing this?",
        "Help me troubleshoot this issue: {issue}",
        "What are common reasons for {symptom} in {system}?",
        "Debug this code: ```{language}\n{code}\n```",
    ],
    "Design": [
        "Help me design a {component} that achieves {goal}.",
        "What's a good design pattern for {situation}?",
        "Suggest a color scheme for a {project_type} with a {mood} feel.",
        "How should I structure the information architecture for a {website_type}?",
        "Create a wireframe concept for a {page_type} page.",
    ],
    "Chat": [
        "How's your day going?",
        "Tell me something interesting that happened recently.",
        "What's your opinion on {topic}?",
        "If you could visit any place, where would you go and why?",
        "Do you have any recommendations for good {media_type} about {subject}?",
    ],
    "Search": [
        "Search for the latest information about {topic}.",
        "Find recent news on {event}.",
        "What are the current developments in {field}?",
        "Search for up-to-date information on {subject}.",
        "What does the internet say about {topic} right now?",
    ],
    "SeeImage": [
        "Analyze this image and tell me what you see: [image]",
        "Describe what's happening in this picture: [image]",
        "What can you tell me about this image? [image]",
        "Identify the objects and elements in this image: [image]",
        "Can you explain what's shown in this visual? [image]",
    ],
}


async def set_commands(
    use_all: bool = True, custom_commands: List[Dict] = None
) -> None:
    """
    Set available commands in the Chainlit UI.

    Args:
            use_all: Whether to use all predefined commands or not
            custom_commands: Optional list of custom command dictionaries to use instead
    """
    commands_to_set = []

    if custom_commands:
        commands_to_set = custom_commands
    elif use_all:
        # Use a cleaned version of commands (without the route attribute which is for internal use)
        commands_to_set = [
            {k: v for k, v in cmd.items() if k != "route"} for cmd in COMMANDS
        ]
    else:
        # Use a subset of commands if not using all
        available_commands = COMMANDS.copy()
        shuffle(available_commands)
        subset = available_commands[:4]  # Limit to 4 random commands
        commands_to_set = [
            {k: v for k, v in cmd.items() if k != "route"} for cmd in subset
        ]

    await cl.context.emitter.set_commands(commands_to_set)


def get_command_template(command_id: str) -> Optional[str]:
    """
    Get a random template for the specified command.

    Args:
            command_id: The ID of the command

    Returns:
            Optional[str]: A template string or None if command_id is not found
    """
    templates = COMMAND_TEMPLATES.get(command_id)
    if not templates:
        return None

    return choice(templates)


def get_command_route(command_id: str) -> Optional[str]:
    """
    Get the router route associated with a command.

    Args:
            command_id: The ID of the command

    Returns:
            Optional[str]: The route name or None if command_id is not found
    """
    for command in COMMANDS:
        if command["id"] == command_id:
            return command.get("route")
    return None


# Route-based command generation
def build_commands_from_routes(max_count: int = 5) -> List[Dict]:
    """
    Build commands from router routes.

    Args:
            max_count: Maximum number of commands to return

    Returns:
            List[Dict]: List of command dictionaries
    """
    # Get all routes from the router
    routes = create_routes()

    # Create mapping for route commands
    route_commands = {
        "text-processing": {
            "id": "Analyze",
            "icon": "file-text",
            "description": "Analyze text content",
            "route": SemanticRouterType.TEXT_PROCESSING,
        },
        "vision-image-processing": {
            "id": "SeeImage",
            "icon": "eye",
            "description": "Analyze images",
            "route": SemanticRouterType.VISION_IMAGE_PROCESSING,
        },
        "casual-conversation": {
            "id": "Chat",
            "icon": "message-circle",
            "description": "Have a conversation",
            "route": SemanticRouterType.CASUAL_CONVERSATION,
        },
        "image-generation": {
            "id": "CreateImage",
            "icon": "image-plus",
            "description": "Generate images",
            "route": SemanticRouterType.IMAGE_GENERATION,
        },
        "curious": {
            "id": "Learn",
            "icon": "book-open",
            "description": "Learn about a topic",
            "route": SemanticRouterType.CURIOUS,
        },
        "web-search": {
            "id": "Search",
            "icon": "search",
            "description": "Search the web",
            "route": SemanticRouterType.WEB_SEARCH,
        },
    }

    # Create commands from routes
    all_commands = []
    route_names = list(route_commands.keys())
    random.shuffle(route_names)

    # Select up to max_count routes
    selected_routes = route_names[:max_count]

    for route_name in selected_routes:
        command = route_commands.get(route_name)
        if command:
            # Add button and persistent attributes
            command["button"] = True
            command["persistent"] = False
            # Clone the command and remove the route for external use
            display_command = {k: v for k, v in command.items() if k != "route"}
            all_commands.append(display_command)

    return all_commands


def generate_random_prompt() -> Dict[str, str]:
    """
    Generate a random starter prompt.

    Returns:
        Dict[str, str]: Dictionary with label and message
    """
    # Use a random prompt from the hardcoded list
    if STARTER_PROMPTS:
        return random.choice(STARTER_PROMPTS)

    # Fallback prompt if no hardcoded prompts are available
    return {
        "label": "Default Question",
        "message": "Hello! I'd like to learn more about how large language models work. Can you explain the basic concepts?",
    }


def get_starter_prompts(
    max_count: int = 5, use_llm: bool = False, refresh_cache: bool = False
) -> List[Dict[str, str]]:
    """
    Get a list of starter prompts.

    Args:
        max_count: Maximum number of prompts to return
        use_llm: Whether to use LLM-generated prompts (currently not implemented)
        refresh_cache: Whether to refresh any cached prompts

    Returns:
        List[Dict[str, str]]: List of prompt dictionaries with label and message
    """
    # For now, just use the hardcoded prompts
    # The use_llm and refresh_cache params are for future extension
    prompts = STARTER_PROMPTS.copy()

    # Shuffle the prompts
    random.shuffle(prompts)

    # Return up to max_count prompts
    return prompts[:max_count]


def get_shuffled_starters(
    max_count: int = 5, use_llm: bool = False, use_random: bool = False
) -> List[cl.Starter]:
    """
    Get a shuffled list of starter prompts.

    Args:
        max_count: Maximum number of starters to return
        use_llm: Whether to use LLM-generated starters
        use_random: Whether to use a single random starter instead of multiple

    Returns:
        List[cl.Starter]: List of Chainlit Starter objects
    """
    if use_random:
        # Return a single random starter
        all_prompts = get_starter_prompts(
            max_count=1, use_llm=use_llm, refresh_cache=False
        )
        starter = (
            all_prompts[0]
            if all_prompts
            else {"label": "Default", "message": "Hello! How can I help you today?"}
        )
        return [cl.Starter(label=starter["label"], message=starter["message"])]

    # Get all available starters
    all_prompts = get_starter_prompts(
        max_count=max_count, use_llm=use_llm, refresh_cache=False
    )

    # Shuffle the starters
    random.shuffle(all_prompts)

    # Limit to max_count
    selected_prompts = all_prompts[:max_count]

    # Convert to Chainlit Starter objects
    return [
        cl.Starter(label=item["label"], message=item["message"])
        for item in selected_prompts
    ]
