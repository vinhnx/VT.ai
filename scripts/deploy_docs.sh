#!/bin/bash

# Set the working directory to the VT.ai project root directory
cd "$(dirname "$0")/.." || { echo "Error: Cannot find VT.ai root directory"; exit 1; }

# First, run the build script to generate the site and copy llms.txt
./scripts/build_docs.sh

# If the build was successful, deploy to GitHub Pages using mkdocs
if [ $? -eq 0 ]; then
    echo "Building was successful, deploying to GitHub Pages..."

    # Use mkdocs gh-deploy to deploy to GitHub Pages
    # This will build and push to the gh-pages branch in one step
    mkdocs gh-deploy --force --clean --message "Deploy documentation with llms.txt support"

    echo "Deployment complete! Your documentation should be available at https://vinhnx.github.io/VT.ai/"
else
    echo "Build failed, not deploying."
fi