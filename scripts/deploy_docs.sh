# Set the working directory to the VT.ai project root directory
cd "$(dirname "$0")/.." || { echo "Error: Cannot find VT.ai root directory"; exit 1; }
#!/bin/bash

# First, run the build script to generate the site and copy llms.txt
./scripts/build_docs.sh

# If the build was successful, deploy to GitHub Pages
if [ $? -eq 0 ]; then
    echo "Building was successful, deploying to GitHub Pages..."

    # Create a temporary directory for the gh-pages branch
    mkdir -p temp_deploy
    cd temp_deploy

    # Initialize a git repo and set remote
    git init
    git remote add origin git@github.com:vinhnx/VT.ai.git
    git checkout -b gh-pages

    # Copy the site contents
    cp -r ../site/* .
    cp ../site/.nojekyll .

    # Add, commit, and push
    git add --all
    git commit -m "Deploy documentation with llms.txt support"
    git push -f origin gh-pages

    # Clean up
    cd ..
    rm -rf temp_deploy

    echo "Deployment complete! Your documentation should be available at https://vinhnx.github.io/VT.ai/"
else
    echo "Build failed, not deploying."
fi