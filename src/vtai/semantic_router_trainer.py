from semantic_router import Route
from semantic_router.encoders import FastEmbedEncoder
from semantic_router.layer import RouteLayer

# usage:
# ```
# $ python src/vtai/semantic_router_trainer.py
# ```
# then semantic_route_layers.json file will be updated

routes = [
    Route(
        name="vision",
        utterances=[
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
        name="chitchat",
        utterances=[
            "how's the weather today?",
            "how are things going?",
            "lovely weather today",
            "the weather is horrendous",
            "let's go to the chippy",
        ],
    ),
    Route(
        name="image-gen",
        utterances=[
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
]

encoder = FastEmbedEncoder(name="BAAI/bge-small-en-v1.5")
layer = RouteLayer(encoder=encoder, routes=routes)
layer.to_json("semantic_route_layers.json")
