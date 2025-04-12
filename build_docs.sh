#!/bin/bash

# Build the MkDocs site
mkdocs build

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "MkDocs build successful, copying llms.txt to site directory..."
    cp llms.txt site/
    echo "llms.txt copied successfully."
else
    echo "MkDocs build failed. llms.txt was not copied."
fi