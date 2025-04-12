#!/bin/bash

# Build the MkDocs site
mkdocs build

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "MkDocs build successful, copying llms.txt to site directory..."
    cp llms.txt site/
    # Create .nojekyll file in site directory to prevent GitHub Pages from using Jekyll
    touch site/.nojekyll
    echo "llms.txt and .nojekyll copied successfully."
else
    echo "MkDocs build failed. llms.txt was not copied."
fi