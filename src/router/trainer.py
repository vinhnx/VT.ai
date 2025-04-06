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
    # Route definitions remain unchanged...
    routes = [
        Route(
            name="text-processing",
            utterances=[
                # utterances remain unchanged...
            ],
        ),
        # other routes remain unchanged...
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

    args = parser.parse_args()

    try:
        routes = create_routes()
        # Initialize FastEmbedEncoder explicitly without falling back to OpenAI
        encoder = FastEmbedEncoder(model_name="BAAI/bge-small-en-v1.5")

        print("Training semantic router using FastEmbedEncoder...")
        layer = RouteLayer(encoder=encoder, routes=routes)

        # Save the trained layer
        output_path = args.output
        layer.to_json(output_path)
        print(f"Successfully saved semantic router layer to {output_path}")

    except Exception as e:
        print(f"Error training semantic router: {str(e)}")
        return


if __name__ == "__main__":
    main()
