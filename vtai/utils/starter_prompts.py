"""
Starter prompt utilities for VT.ai application.

This module provides a collection of high-quality hardcoded starter prompts
and functions to randomly select from them. It also includes the ability to
generate route-based starter prompts as an alternative approach.
"""

import random
from random import choice, shuffle
from typing import Dict, List

import chainlit as cl

from vtai.router.trainer import create_routes

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


def generate_random_prompt() -> Dict[str, str]:
    """
    Select a random prompt from the hardcoded STARTER_PROMPTS list.

    Returns:
        Dict[str, str]: A dictionary containing a label and message for a prompt
    """
    return choice(STARTER_PROMPTS)


# Route-based starter prompts
def build_starters_from_routes(max_count: int = 5) -> List[cl.Starter]:
    """
    Build starter prompts from the router routes.
    Each route category will be converted into a starter prompt with a short label and verbose message.

    Args:
        max_count: Maximum number of starters to return

    Returns:
        List[cl.Starter]: List of cl.Starter objects
    """
    # Get all routes from the router
    routes = create_routes()

    # Create mapping of short labels for each route
    route_labels = {
        "text-processing": "Text Analysis",
        "vision-image-processing": "Analyze Image",
        "casual-conversation": "Chat",
        "image-generation": "Create Image",
        "curious": "Tell Me About",
        "code-assistance": "Code Help",
        "data-analysis": "Analyze Data",
        "creative-writing": "Write Something",
        "planning-organization": "Plan This",
        "troubleshooting": "Fix My Issue",
    }

    # Create expanded messages for each route
    route_messages = {}
    for route in routes:
        if route.name in route_labels:
            # Select a random utterance from the route
            utterance = random.choice(route.utterances)

            # Expand the utterance into a more verbose message
            if route.name == "image-generation":
                route_messages[route.name] = (
                    f"I'd like you to {utterance}. Please make it highly detailed with vibrant colors and an interesting composition."
                )

            elif route.name == "code-assistance":
                route_messages[route.name] = (
                    f"{utterance}. I'm looking for clean, efficient code with good documentation. Please explain your reasoning and any best practices you're applying."
                )

            elif route.name == "data-analysis":
                route_messages[route.name] = (
                    f"{utterance}. I'm interested in both the statistical significance and practical implications of the findings. Please include visual representation suggestions if appropriate."
                )

            elif route.name == "creative-writing":
                route_messages[route.name] = (
                    f"{utterance}. I'd like something unique with vivid imagery and compelling character development. Feel free to explore unexpected directions."
                )

            elif route.name == "planning-organization":
                route_messages[route.name] = (
                    f"{utterance}. I'm looking for a comprehensive approach that considers potential obstacles and includes contingency plans. Please make it practical and implementable."
                )

            elif route.name == "troubleshooting":
                route_messages[route.name] = (
                    f"{utterance}. I've already tried restarting and checking basic connectivity. Please provide a step-by-step diagnostic process and potential solutions ranked by likelihood."
                )

            elif route.name == "vision-image-processing":
                route_messages[route.name] = (
                    f"{utterance}. Please provide details about the key elements, composition, color scheme, and any text or symbols present. Also share any insights about the context or purpose of the image."
                )

            elif route.name == "text-processing":
                route_messages[route.name] = (
                    f"{utterance}. I'd like a thorough analysis that covers tone, key arguments, implicit assumptions, and overall effectiveness. Please suggest improvements where appropriate."
                )

            elif route.name == "casual-conversation":
                route_messages[route.name] = (
                    f"{utterance} I'd love to hear your thoughts on this in a conversational way, as if we're just chatting casually."
                )

            elif route.name == "curious":
                route_messages[route.name] = (
                    f"{utterance} Please provide a comprehensive explanation with interesting facts, historical context, and current developments. I'm particularly interested in aspects that might surprise someone new to the topic."
                )

    # Create starters from routes
    all_starters = []
    route_names = list(route_labels.keys())
    # Shuffle to get random selection each time
    random.shuffle(route_names)

    # Select up to max_count routes
    selected_routes = route_names[:max_count]

    for route_name in selected_routes:
        label = route_labels.get(route_name)
        message = route_messages.get(route_name)

        if label and message:
            all_starters.append({"label": label, "message": message})

    # Convert to Chainlit Starter objects
    return [
        cl.Starter(label=item["label"], message=item["message"])
        for item in all_starters
    ]


def get_starter_prompts(
    max_count: int = 5, use_llm: bool = False, refresh_cache: bool = False
) -> List[Dict[str, str]]:
    """
    Get starter prompts using the preferred method.
    The use_llm parameter is kept for API compatibility but ignored since we're using hardcoded prompts.

    Args:
        max_count: Maximum number of prompts to return (minimum 5)
        use_llm: Ignored parameter (kept for API compatibility)
        refresh_cache: Ignored parameter (kept for API compatibility)

    Returns:
        List[Dict[str, str]]: List of prompt dictionaries with 'label' and 'message' keys
    """
    # Ensure at least 5 starter prompts are shown
    max_count = max(5, max_count)

    # Randomly decide whether to use route-based or hardcoded prompts
    use_route_based = random.choice([True, False])

    if use_route_based:
        return [
            {"label": starter.label, "message": starter.message}
            for starter in build_starters_from_routes(max_count=max_count)
        ]
    else:
        # Use hardcoded prompts
        starter_list = STARTER_PROMPTS.copy()
        shuffle(starter_list)
        return starter_list[:max_count]


def get_shuffled_starters(
    max_count: int = 5, use_llm: bool = False, use_random: bool = False
) -> List[cl.Starter]:
    """
    Get shuffled starters for chat profiles.
    The use_llm parameter is kept for API compatibility but ignored since we're using hardcoded prompts.

    Args:
        max_count: Maximum number of starters to return (minimum 5)
        use_llm: Ignored parameter (kept for API compatibility)
        use_random: Whether to use route-based starters

    Returns:
        List[cl.Starter]: List of cl.Starter objects
    """
    # Ensure at least 5 starter prompts are shown
    max_count = max(5, max_count)

    if use_random:
        # Use dynamic route-based starters
        return build_starters_from_routes(max_count=max_count)
    else:
        # Use hardcoded starters
        starters_data = get_starter_prompts(max_count=max_count, use_llm=False)
        return [
            cl.Starter(label=item["label"], message=item["message"])
            for item in starters_data
        ]
