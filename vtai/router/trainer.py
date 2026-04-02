import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import dotenv
from semantic_router import Route, SemanticRouter
from semantic_router.encoders import FastEmbedEncoder

dotenv.load_dotenv(dotenv.find_dotenv())

# usage:
# ```
# $ python vtai/router/trainer.py
# ```
# then layers.json file will be updated


def create_routes() -> list[Route]:
    """
    Create and return predefined routes for the semantic router.
    Each route represents a specific conversation topic with example utterances.
    """
    routes = [
        Route(
            name="text-processing",
            utterances=[
                "Summarize this text for me",
                "Can you extract the key points from this passage?",
                "Provide a summary of this article",
                "Give me a concise overview of this document",
                "Distill the main ideas from this text",
                "Analyze the sentiment of this review",
                "What is the overall tone of this piece?",
                "Detect the emotion expressed in this message",
                "Classify the topic of this text",
                "Categorize this content into a specific domain",
                "Translate this text to Spanish",
                "Convert this document to French",
                "Rephrase this sentence in simpler terms",
                "Simplify this paragraph for easier reading",
                "Explain this technical jargon in plain language",
                "Find and correct any grammar errors in this text",
                "Check this writing for spelling mistakes",
                "Edit this document for style and clarity",
            ],
        ),
        Route(
            name="vision-image-processing",
            utterances=[
                "Explain this image",
                "Explain this image with url",
                "What's in this image?",
                "Describe the contents of this picture",
                "Tell me what you see in this image",
                "Identify the objects and elements in this scene",
                "What's happening in this visual?",
                "Give me a detailed description of this picture",
                "Analyze the components of this image for me",
                "Break down the different parts of this scene",
                "List the key features you notice in this image",
                "Explain what's depicted in this visual representation",
            ],
        ),
        Route(
            name="casual-conversation",
            utterances=[
                "How's your day going?",
                "Did you catch the game last night?",
                "I'm so ready for the weekend",
                "Any fun plans coming up?",
                "The weather has been gorgeous lately",
                "Have you tried that new restaurant downtown?",
                "I've been binge-watching this great new show",
                "I could use a vacation, how about you?",
                "My commute was a nightmare this morning",
                "I'm thinking of picking up a new hobby, any ideas?",
                "Did you hear about the latest celebrity gossip?",
                "I can't believe how quickly the time flies",
                "I'm already counting down to the holidays!",
                "You'll never guess what happened to me today",
                "Do you have any book recommendations?",
            ],
        ),
        Route(
            name="image-generation",
            utterances=[
                "please help me generate a photo of a car",
                "a image of a cute puppy playing in a meadow",
                "a image of a majestic mountain landscape at sunset",
                "a photo of a modern city skyline at night",
                "generate a image of a tropical beach with palm trees",
                "generate another image of a futuristic spacecraft",
                "a image of a cozy cabin in the woods during winter",
                "a photo of a vintage car from the 1950s",
                "generate a image of an underwater coral reef scene",
                "generate another image of a fantastical creature from mythology",
                "a image of a serene Japanese zen garden",
                "a photo of a delicious gourmet meal on a plate",
                "generate a image of a whimsical treehouse in a magical forest",
                "generate another image of an abstract painting with bold colors",
                "a image of a historic castle on a hilltop",
                "a photo of a beautiful flower bouquet in a vase",
                "generate a image of a sleek and modern sports car",
                "generate another image of a breathtaking northern lights display",
                "a image of a cozy reading nook with bookshelves",
                "a photo of a stunning desert landscape at sunset",
                "generate a image of a realistic portrait of a person",
                "a image of a playful kitten chasing a toy",
                "a photo of a golden retriever puppy with a wagging tail",
                "generate a image of a majestic lion in the savannah",
                "generate another image of a school of tropical fish swimming",
                "a image of a fluffy white rabbit in a garden",
                "a photo of a parrot with colorful feathers",
                "generate a image of a curious owl perched on a branch",
                "generate another image of a dolphin jumping out of the water",
                "a image of a tortoise basking in the sun",
                "a photo of a horse running through a grassy field",
                "a image of a cute kitten sleeping in a basket",
                "a photo of a playful dog chasing a frisbee in the park",
                "generate a image of a cat lounging on a sunny windowsill",
                "generate another image of a dog fetching a stick in the water",
                "a image of a kitten playing with a ball of yarn",
                "a photo of a dog begging for treats with puppy eyes",
                "generate a image of a cat stretching on a cozy couch",
                "generate another image of a dog running on the beach",
                "a image of a cat grooming itself with its paw",
                "a photo of a dog wearing a colorful bandana",
            ],
        ),
        Route(
            name="curious",
            utterances=[
                "Tell me something interesting",
                "What's something fascinating you've learned recently?",
                "Share an intriguing fact with me",
                "I'm curious to know more about...",
                "Enlighten me on the topic of...",
                "Can you explain the concept of...?",
                "I've always wondered about...",
                "How does ... work?",
                "What's the story behind...?",
                "I'm really interested in learning about...",
                "Teach me something new today",
                "Give me a fun trivia fact",
                "What's the most mind-blowing thing you know?",
                "Hit me with some knowledge I don't have yet",
                "I love learning new things, what can you tell me?",
            ],
        ),
        Route(
            name="web-search",
            utterances=[
                "Who is",
                "Search for information about",
                "Find the latest news on",
                "Look up information about",
                "Get me up-to-date information on",
                "Search the web for",
                "What are the recent developments in",
                "Find current information about",
                "What's happening with",
                "Look online for information about",
                "I need recent information about",
                "Get me the latest details on",
                "Find out what's new with",
                "What does the internet say about",
                "Search for recent news about",
                "Can you find information on",
                "Look up the current status of",
                "What's the latest on",
                "Find me up-to-date facts about",
                "Get real-time information about",
                "Tell me what's currently happening with",
            ],
        ),
    ]
    return routes


def load_routes_from_file(file_path: str) -> List[Route]:
    """
    Load additional routes from a JSON file.

    Args:
        file_path: Path to the JSON file containing route definitions

    Returns:
        List of Route objects
    """
    routes = []
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        for route_data in data.get('routes', []):
            route = Route(
                name=route_data['name'],
                utterances=route_data.get('utterances', [])
            )
            routes.append(route)

        print(f"Loaded {len(routes)} routes from {file_path}")
    except FileNotFoundError:
        print(f"Route file not found: {file_path}")
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in route file: {e}")
    except Exception as e:
        print(f"Error loading routes: {e}")

    return routes


def analyze_route_coverage(routes: List[Route]) -> Dict:
    """
    Analyze the coverage and balance of routes.

    Args:
        routes: List of routes to analyze

    Returns:
        Dictionary with analysis results
    """
    analysis = {
        "total_routes": len(routes),
        "total_utterances": sum(len(r.utterances) for r in routes),
        "avg_utterances_per_route": 0,
        "min_utterances": 0,
        "max_utterances": 0,
        "route_details": []
    }

    if not routes:
        return analysis

    utterance_counts = [len(r.utterances) for r in routes]
    analysis["avg_utterances_per_route"] = sum(utterance_counts) / len(utterance_counts)
    analysis["min_utterances"] = min(utterance_counts)
    analysis["max_utterances"] = max(utterance_counts)

    for route in routes:
        analysis["route_details"].append({
            "name": route.name,
            "utterance_count": len(route.utterances),
            "avg_word_count": sum(len(u.split()) for u in route.utterances) / len(route.utterances) if route.utterances else 0
        })

    return analysis


def train_router(
    routes: List[Route],
    model_name: str = "BAAI/bge-small-en-v1.5",
    output_path: str = "./vtai/router/layers.json",
    verbose: bool = False
) -> SemanticRouter:
    """
    Train the semantic router with given routes.

    Args:
        routes: List of Route objects to train with
        model_name: Name of the encoder model to use
        output_path: Path to save the trained layer
        verbose: Enable verbose output

    Returns:
        Trained SemanticRouter object
    """
    start_time = time.time()

    # Initialize FastEmbedEncoder explicitly without falling back to OpenAI
    encoder = FastEmbedEncoder(model_name=model_name)

    if verbose:
        print(f"Training semantic router using FastEmbedEncoder: {model_name}")
        print(f"Created layer with {len(routes)} routes:")
        for i, route in enumerate(routes):
            print(f"  {i + 1}. '{route.name}' - {len(route.utterances)} utterances")

    layer = SemanticRouter(encoder=encoder, routes=routes)

    # Save the trained layer
    layer.to_json(output_path)

    elapsed = time.time() - start_time

    if verbose:
        print(f"Successfully saved semantic router layer to {output_path}")
        print(f"Training completed in {elapsed:.2f} seconds")

    return layer


def get_model_options() -> List[str]:
    """
    Get available encoder model options.

    Returns:
        List of model names
    """
    return [
        "BAAI/bge-small-en-v1.5",  # Fast, good quality
        "BAAI/bge-base-en-v1.5",   # Better quality, slower
        "BAAI/bge-large-en-v1.5",  # Best quality, slowest
        "sentence-transformers/all-MiniLM-L6-v2",  # Alternative
    ]


def main() -> None:
    """
    Main function to train and save the semantic router.
    """
    parser = argparse.ArgumentParser(description="Train semantic router")
    parser.add_argument(
        "--output",
        type=str,
        default="./vtai/router/layers.json",
        help="Path to save the trained layer",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output during training",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        choices=get_model_options(),
        help="Encoder model to use",
    )
    parser.add_argument(
        "--analyze",
        "-a",
        action="store_true",
        help="Analyze route coverage without training",
    )
    parser.add_argument(
        "--add-routes",
        type=str,
        help="Path to JSON file with additional routes to add",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics after training",
    )

    args = parser.parse_args()

    try:
        routes = create_routes()

        # Load additional routes if provided
        if args.add_routes:
            additional_routes = load_routes_from_file(args.add_routes)
            routes.extend(additional_routes)

        # Analyze routes if requested
        if args.analyze:
            analysis = analyze_route_coverage(routes)
            print("\n=== Route Coverage Analysis ===")
            print(f"Total routes: {analysis['total_routes']}")
            print(f"Total utterances: {analysis['total_utterances']}")
            print(f"Average utterances per route: {analysis['avg_utterances_per_route']:.1f}")
            print(f"Min utterances: {analysis['min_utterances']}")
            print(f"Max utterances: {analysis['max_utterances']}")
            print("\nPer-route details:")
            for detail in analysis['route_details']:
                print(f"  {detail['name']}: {detail['utterance_count']} utterances, "
                      f"avg {detail['avg_word_count']:.1f} words")
            return

        # Train the router
        layer = train_router(
            routes=routes,
            model_name=args.model,
            output_path=args.output,
            verbose=args.verbose
        )

        # Show statistics if requested
        if args.stats:
            analysis = analyze_route_coverage(routes)
            print("\n=== Training Statistics ===")
            print(f"Model: {args.model}")
            print(f"Total routes: {analysis['total_routes']}")
            print(f"Total utterances: {analysis['total_utterances']}")
            print(f"Output: {args.output}")

    except Exception as e:
        print(f"Error training semantic router: {str(e)}")
        return


if __name__ == "__main__":
    main()
