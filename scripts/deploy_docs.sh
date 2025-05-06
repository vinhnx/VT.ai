#!/bin/bash

# Set the working directory to the VT.ai project root directory
cd "$(dirname "$0")/.." || { echo "Error: Cannot find VT.ai root directory"; exit 1; }

# First, run the build script to generate the site and copy llms.txt
./scripts/build_docs.sh

# If the build was successful, deploy to GitHub Pages using mkdocs
if [ $? -eq 0 ]; then
	echo "Building was successful, preparing to deploy to GitHub Pages..."
	
	# Add llms.txt to git explicitly to ensure it's tracked
	if [ -f "site/llms.txt" ]; then
		echo "Ensuring llms.txt is tracked by git..."
		git add site/llms.txt -f
	else
		echo "Warning: llms.txt not found in site directory."
		echo "Would you like to create and deploy it now? (y/n)"
		read -r create_deploy
		if [ "$create_deploy" = "y" ]; then
			./scripts/deploy_llms_txt.py --create-if-missing --force-build
			# Exit early since deploy_llms_txt.py handles the deployment
			exit $?
		fi
	fi
	
	# Create .nojekyll file to prevent GitHub Pages from using Jekyll
	if [ ! -f "site/.nojekyll" ]; then
		echo "Creating .nojekyll file..."
		touch site/.nojekyll
		git add site/.nojekyll -f
	fi
	
	# Use mkdocs gh-deploy to deploy to GitHub Pages
	echo "Deploying to GitHub Pages..."
	mkdocs gh-deploy --force --no-jekyll --message "Deploy documentation with llms.txt support"
	
	# Verify deployment
	echo "Deployment complete! Verifying llms.txt..."
	echo "Your documentation should be available at https://vinhnx.github.io/VT.ai/"
	echo "llms.txt should be available at https://vinhnx.github.io/VT.ai/llms.txt"
	
	# Optional: Check if the llms.txt file is accessible (requires curl)
	if command -v curl &> /dev/null; then
		echo "Waiting 20 seconds for GitHub Pages to update..."
		sleep 20
		
		echo "Checking if llms.txt is accessible..."
		if curl --output /dev/null --silent --head --fail "https://vinhnx.github.io/VT.ai/llms.txt"; then
			echo "✅ Success! llms.txt is accessible."
		else
			echo "⚠️ Warning: Could not access llms.txt. It may take a few minutes for GitHub Pages to update."
			echo "You can verify manually in a few minutes, or run the deploy_llms_txt.py script with verification:"
			echo "  ./scripts/deploy_llms_txt.py --verify --attempts 5 --wait 30"
		fi
	else
		echo "curl not available. Cannot verify deployment automatically."
		echo "Please check manually if llms.txt is accessible at https://vinhnx.github.io/VT.ai/llms.txt after a few minutes."
	fi
else
	echo "Build failed, not deploying."
	exit 1
fi