import argparse

import dotenv
from semantic_router import Route
from semantic_router.encoders import FastEmbedEncoder
from semantic_router.layer import RouteLayer

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
    ]
    return routes


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

    args = parser.parse_args()

    try:
        routes = create_routes()
        # Initialize FastEmbedEncoder explicitly without falling back to OpenAI
        model_name = "BAAI/bge-small-en-v1.5"
        encoder = FastEmbedEncoder(model_name=model_name)

        print("Training semantic router using FastEmbedEncoder...")
        layer = RouteLayer(encoder=encoder, routes=routes)

        if args.verbose:
            print(f"Created layer with {len(routes)} routes:")
            for i, route in enumerate(routes):
                print(f"  {i+1}. '{route.name}' - {len(route.utterances)} utterances")
            # Fixed: Access the model name from the variable instead of the attribute
            print(f"Encoder: {model_name}")

        # Save the trained layer
        output_path = args.output
        layer.to_json(output_path)
        print(f"Successfully saved semantic router layer to {output_path}")

    except Exception as e:
        print(f"Error training semantic router: {str(e)}")
        return


if __name__ == "__main__":
    main()
