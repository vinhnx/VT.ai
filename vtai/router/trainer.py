import argparse

import dotenv
from semantic_router import Route
from semantic_router.encoders import FastEmbedEncoder
from semantic_router.layer import RouteLayer

dotenv.load_dotenv(dotenv.find_dotenv())

# usage:
# ```
# $ python vtai/router/trainer.py --verbose
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
                "Can you proofread this essay for me?",
                "Identify the key themes in this article",
                "What's the main argument in this paper?",
                "Make this text more formal for a business audience",
                "Rewrite this in a more casual tone",
                "Extract all the dates and numbers from this document",
                "Can you identify any bias in this news article?",
                "Condense this 3-page report into one paragraph",
                "Find all the technical terms in this document and explain them",
                "Compare the writing styles of these two passages",
                "Suggest ways to improve the clarity of this text",
                "Format this content for a blog post",
                "Help me improve the flow of this paragraph",
                "Convert this academic text into something a 10-year-old would understand",
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
                "Is there any text in this image? What does it say?",
                "What colors are predominantly used in this picture?",
                "Describe the emotions conveyed by this image",
                "Does this image contain any people? What are they doing?",
                "What time of day does this photo appear to be taken?",
                "Compare what's in the foreground versus the background of this image",
                "Analyze the composition of this photograph",
                "Is there anything unusual or out of place in this picture?",
                "What style of art would you categorize this image as?",
                "Can you identify any brands or logos in this image?",
                "What's the main focal point of this picture?",
                "Does this image tell a story? What narrative do you see?",
                "Describe the setting or environment shown in this image",
                "What historical period does this image represent?",
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
                "It's been a while since we talked, how have you been?",
                "What's your opinion on the current political situation?",
                "I'm feeling a bit down today",
                "Just wanted to say hello and check in",
                "Do you prefer coffee or tea?",
                "What kind of music do you enjoy?",
                "Have you seen any good movies lately?",
                "What's your favorite season of the year?",
                "I just got a new pet, want to hear about it?",
                "What do you think about social media these days?",
                "I'm thinking about redecorating my apartment",
                "Do you follow any sports teams?",
                "What's your idea of a perfect day?",
                "Have you ever traveled abroad? Where did you go?",
                "What are your thoughts on artificial intelligence?",
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
                "Create an image of a cyberpunk city with neon lights",
                "Draw a picture of a superhero in a dynamic pose",
                "I need an illustration of a fantasy world with floating islands",
                "Could you create a digital painting of a surreal landscape?",
                "Make an image of a steampunk-inspired mechanical creature",
                "Design a logo with a minimalist aesthetic",
                "Generate artwork of a post-apocalyptic scene",
                "Create a photorealistic image of a futuristic vehicle",
                "Draw a hyper-realistic portrait with dramatic lighting",
                "Make an image in the style of impressionist art",
                "I'd like a cartoon-style image of an anthropomorphic animal",
                "Generate an image that visualizes a complex scientific concept",
                "Create a picture that tells a story in one scene",
                "Make an image that combines elements of different art movements",
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
                "What are the latest scientific discoveries?",
                "Tell me about the origins of human language",
                "How do black holes work?",
                "What's the history of chocolate?",
                "Why do we dream when we sleep?",
                "Explain quantum computing in simple terms",
                "What's something most people believe that isn't true?",
                "How did ancient civilizations navigate without GPS?",
                "What causes the northern lights?",
                "Tell me about the most interesting animal adaptations",
                "How does the internet actually work?",
                "What's the science behind déjà vu?",
                "What are the unsolved mysteries in mathematics?",
                "How do vaccines help our immune system?",
                "Why do different languages have different grammar structures?",
            ],
        ),
        Route(
            name="code-assistance",
            utterances=[
                "Help me debug this code",
                "Can you explain what this function does?",
                "Write a script to automate this task",
                "How do I implement this algorithm?",
                "What's wrong with my JavaScript code?",
                "Convert this Python code to Java",
                "Explain this regex pattern",
                "How do I optimize this database query?",
                "Write a unit test for this method",
                "What design pattern should I use for this problem?",
                "How do I fix this error in my code?",
                "Create a REST API endpoint for user authentication",
                "Explain the difference between these two programming approaches",
                "How can I make this code more maintainable?",
                "Generate documentation for this class",
                "How do I use async/await in JavaScript?",
                "What's the best way to handle errors in this function?",
                "Compare these two algorithms for efficiency",
                "Show me how to implement pagination in this API",
                "Refactor this code to use dependency injection",
            ],
        ),
        Route(
            name="data-analysis",
            utterances=[
                "Analyze this dataset for trends",
                "What insights can you derive from these numbers?",
                "Help me interpret this chart",
                "What statistical method should I use for this analysis?",
                "How do I visualize this correlation?",
                "Calculate the growth rate from these figures",
                "Find patterns in this time series data",
                "Compare these two datasets and highlight differences",
                "What's the significance of this p-value?",
                "How reliable is this statistical conclusion?",
                "Help me create a predictive model using this data",
                "What are the key metrics I should track?",
                "Explain the methodology behind this analysis",
                "How do I clean this dataset for better results?",
                "What does this confusion matrix tell me?",
                "Suggest the best chart type for visualizing this data",
                "Help me interpret the results of this A/B test",
                "What can we infer from this sample size?",
                "Explain how to run regression analysis on this data",
                "What biases might be present in this dataset?",
            ],
        ),
        Route(
            name="creative-writing",
            utterances=[
                "Write a short story about a time traveler",
                "Help me come up with a catchy headline for this article",
                "Create a poem about the changing seasons",
                "Write a product description for this new gadget",
                "Generate an engaging introduction for my blog post",
                "Help me write a compelling elevator pitch",
                "Create a character sketch for a fantasy novel",
                "Write a scene with dialogue between two characters",
                "Generate ideas for a creative marketing campaign",
                "Help me craft a persuasive call to action",
                "Write a metaphor that explains this complex concept",
                "Create an alternate ending to this story",
                "Generate a plot outline for a mystery novel",
                "Write a humorous anecdote about everyday life",
                "Help me develop the setting for this narrative",
                "Create taglines for this new brand",
                "Write lyrics for a song about this theme",
                "Help me develop a unique voice for this character",
                "Create a list of potential titles for my book",
                "Write a convincing testimonial for this service",
            ],
        ),
        Route(
            name="planning-organization",
            utterances=[
                "Help me create a schedule for this project",
                "What's the most efficient way to organize these tasks?",
                "Create a to-do list for planning a wedding",
                "How should I prioritize these competing deadlines?",
                "Generate a weekly meal plan with a shopping list",
                "Help me design a study schedule for my exams",
                "What's a good system for organizing digital files?",
                "Create a timeline for launching a new product",
                "How do I track progress on multiple projects?",
                "Help me plan an efficient travel itinerary",
                "What's the best way to delegate these responsibilities?",
                "Create a workout routine that fits my schedule",
                "How do I structure my day for maximum productivity?",
                "Help me organize a virtual team-building event",
                "What's a good framework for setting annual goals?",
                "Create a budget plan for this home renovation project",
                "How should I prepare for this upcoming presentation?",
                "Help me design an onboarding process for new employees",
                "What's a sustainable routine for maintaining work-life balance?",
                "Create a checklist for moving to a new apartment",
            ],
        ),
        Route(
            name="troubleshooting",
            utterances=[
                "Why won't my computer connect to WiFi?",
                "How do I fix this error message on my phone?",
                "My printer isn't working, what should I check?",
                "Troubleshoot why my app keeps crashing",
                "Why is my battery draining so quickly?",
                "My website is loading slowly, how can I fix it?",
                "Diagnose why my smart home devices aren't connecting",
                "How do I resolve this network connection issue?",
                "Why is my screen flickering and how do I stop it?",
                "My microphone isn't being detected, what should I do?",
                "How do I fix audio problems on my video calls?",
                "Troubleshoot why my streaming service keeps buffering",
                "My external hard drive isn't being recognized, what's wrong?",
                "Why am I getting this specific error code on my device?",
                "My email isn't syncing properly, how do I fix it?",
                "Diagnose why my smart TV apps are freezing",
                "What's causing this strange noise in my laptop?",
                "Why won't my bluetooth devices pair anymore?",
                "How do I fix permission issues with this software?",
                "Troubleshoot why this webpage isn't loading correctly",
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
