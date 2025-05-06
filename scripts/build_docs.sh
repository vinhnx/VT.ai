#!/bin/bash

# Set the working directory to the VT.ai project root directory
cd "$(dirname "$0")/.." || { echo "Error: Cannot find VT.ai root directory"; exit 1; }

# Create the site directory if it doesn't exist
mkdir -p site

# Ensure the llms.txt file exists in the project root
if [ ! -f "llms.txt" ]; then
	echo "Warning: llms.txt not found in project root."
	echo "Would you like to create a basic llms.txt file? (y/n)"
	read -r create_llms_txt
	if [ "$create_llms_txt" = "y" ]; then
		python scripts/deploy_llms_txt.py --create-if-missing
		if [ ! -f "llms.txt" ]; then
			echo "Error: Failed to create llms.txt file."
			exit 1
		fi
	else
		echo "Continuing without llms.txt file..."
	fi
fi

# First, copy llms.txt directly to the site directory (before building)
if [ -f "llms.txt" ]; then
	echo "Copying llms.txt to site directory..."
	cp llms.txt site/
	echo "llms.txt copied successfully."
fi

# Create .nojekyll file in site directory to prevent GitHub Pages from using Jekyll
touch site/.nojekyll

# Build the MkDocs site
echo "Building documentation with MkDocs..."
mkdocs build

# Verify the llms.txt file exists in the site directory
if [ -f "llms.txt" ] && [ ! -f "site/llms.txt" ]; then
	echo "Warning: llms.txt not found in site directory after build. Copying again..."
	cp llms.txt site/
	echo "llms.txt copied successfully."
elif [ -f "site/llms.txt" ]; then
	echo "✅ Verified: llms.txt exists in the site directory."
fi

# Make sure .nojekyll file exists (it might have been removed during the build)
if [ ! -f "site/.nojekyll" ]; then
	echo "Creating .nojekyll file to prevent GitHub Pages from using Jekyll..."
	touch site/.nojekyll
fi

echo "Documentation build completed. Ready for deployment."