#!/usr/bin/env python3
"""
VT.ai - Release automation script
This script handles:
1. Version bumping in pyproject.toml
2. Optional git tagging
3. Building distribution packages
4. Uploading to PyPI
5. Documentation building and deployment with MkDocs
6. Optional pushing of tags and changes to GitHub
7. Deploying llms.txt to GitHub Pages
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(command, description=None):
    """Run a shell command and print its output"""
    print(f"\n{'=' * 50}")
    if description:
        print(f"{description}...")
    print(f"Running: {command}")
    print("=" * 50)

    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)

    return result.stdout.strip()


def check_command(command):
    """Run a shell command and return output without printing, return None if command fails"""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def get_current_version():
    """Get the current version from pyproject.toml"""
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()
    version_match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if version_match:
        return version_match.group(1)
    raise ValueError("Could not find version in pyproject.toml")


def update_version(new_version):
    """Update the version in pyproject.toml"""
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()
    updated_content = re.sub(
        r'version\s*=\s*"([^"]+)"', f'version = "{new_version}"', content
    )
    pyproject_path.write_text(updated_content)
    print(f"Version updated to {new_version} in pyproject.toml")


def tag_exists(tag_name):
    """Check if a git tag already exists"""
    result = check_command(f"git tag -l {tag_name}")
    return result and tag_name in result.splitlines()


def check_mkdocs_installation():
    """Check if MkDocs is properly installed"""
    mkdocs_version = check_command("mkdocs --version")
    if not mkdocs_version:
        print("⚠️  MkDocs is not installed or not in the PATH")
        install = input("Install MkDocs using uv? (y/n): ").lower()
        if install == "y":
            run_command(
                "uv pip install -U mkdocs mkdocs-material mkdocstrings mkdocstrings-python",
                "Installing MkDocs and required extensions",
            )
            return True
        return False

    print(f"Found MkDocs: {mkdocs_version}")
    return True


def validate_mkdocs_config():
    """Validate the MkDocs configuration file"""
    if not Path("mkdocs.yml").exists():
        print("⚠️  mkdocs.yml not found in the current directory")
        return False

    result = check_command("mkdocs build --strict --dry-run")
    if not result:
        print("⚠️  MkDocs configuration validation failed")
        fix = input("Attempt to continue anyway? (y/n): ").lower()
        return fix == "y"

    return True


def build_docs():
    """Build documentation using MkDocs"""
    if not check_mkdocs_installation():
        return False

    if not validate_mkdocs_config():
        return False

    print("\nBuilding documentation with MkDocs...")
    run_command("mkdocs build --clean", "Building documentation")

    # Ensure llms.txt is copied to the site directory
    if Path("llms.txt").exists():
        print("\nCopying llms.txt to site directory...")
        shutil.copy("llms.txt", "site/")
        print("Creating .nojekyll file to prevent GitHub Pages from using Jekyll...")
        Path("site/.nojekyll").touch()
        print("✅ llms.txt copied successfully to site directory")
    else:
        print("⚠️  llms.txt not found in the project root")
        create_llms = input("Create a basic llms.txt file? (y/n): ").lower()
        if create_llms == "y":
            create_llms_txt_file()
            if Path("llms.txt").exists():
                shutil.copy("llms.txt", "site/")
                print("✅ llms.txt created and copied to site directory")

    return True


def deploy_docs():
    """Deploy documentation to GitHub Pages using MkDocs"""
    if not check_mkdocs_installation():
        return False

    print("\nDeploying documentation to GitHub Pages...")

    # Before deploying, ensure llms.txt is in the site directory and tracked by git
    if Path("site/llms.txt").exists():
        print("Ensuring llms.txt is tracked by git...")
        check_command("git add -f site/llms.txt")

    # Deploy to GitHub Pages with --no-jekyll flag
    run_command(
        "mkdocs gh-deploy --force --no-jekyll", "Deploying documentation with llms.txt"
    )

    # Verify the deployment
    verify_llms_txt_deployment()

    return True


def create_llms_txt_file():
    """Create a basic llms.txt file in the project root"""
    print("\nCreating a basic llms.txt file...")

    repo_info = check_command("git remote get-url origin")
    if repo_info and "github.com" in repo_info:
        repo_url = repo_info.strip()
        if repo_url.endswith(".git"):
            repo_url = repo_url[:-4]
        if repo_url.startswith("git@github.com:"):
            repo_url = repo_url.replace("git@github.com:", "https://github.com/")
    else:
        repo_url = "https://github.com/vinhnx/VT.ai"

    project_name = repo_url.split("/")[-1]

    llms_content = f"""# {project_name} llms.txt
#
# This is the llms.txt file for {project_name}.
# For information about llms.txt files, see https://llmstxt.org/.

title: {project_name} - Documentation
url: {repo_url}
language: en
geography: global
author: vinhnx
modality: documentation

summary: Documentation for {project_name}.

description:
  This is the official documentation for {project_name}, covering installation,
  configuration, usage, and development information.

audience: Users and developers interested in using or extending {project_name}

default-purpose: answer-questions-about-content

documentation:
  - User guide with installation and configuration instructions
  - Developer documentation with architecture and extension details
  - API reference for programmatic usage

key-concepts:
  - AI Integration: Support for various AI models and providers
  - Documentation: Comprehensive guides for users and developers
  - Extension: Ways to customize and extend the functionality

code-of-conduct:
  - Provide accurate information from the documentation
  - Maintain the technical tone of the documentation
  - When answering configuration questions, provide specific examples
  - Highlight best practices mentioned in the documentation

allowed-information:
  - All information in the {project_name} documentation is public
  - Installation and setup procedures
  - Configuration options and APIs
  - Development guidelines and architecture

prohibited-information:
  - Do not provide API keys or security credentials
  - Do not speculate on features not mentioned in the documentation

response-guidelines:
  - For user questions, focus on the practical usage information
  - For developer questions, provide specific API details and examples
  - Include relevant code snippets when appropriate
  - Refer to specific documentation sections when possible

content-structure:
  - User Guide: Getting started, features, configuration, models, troubleshooting
  - Developer Guide: Architecture, extending the application, semantic routing
  - API Reference: Application components and interfaces

versions:
  - The documentation represents the current state of {project_name}
  - Check the GitHub repository for the most up-to-date information

updates:
  - This llms.txt was created on {check_command("date +%Y-%m-%d") or "2025-05-06"}
  - Updates to the documentation can be found in the GitHub repository
"""

    with open("llms.txt", "w") as f:
        f.write(llms_content)

    print("✅ Basic llms.txt file created at the project root")


def verify_llms_txt_deployment():
    """Verify that llms.txt is accessible at the GitHub Pages URL"""
    import time

    # Try to get the GitHub Pages URL from git remote
    repo_info = check_command("git remote get-url origin")
    if repo_info and "github.com" in repo_info:
        # Extract username and repo name from remote URL
        if "github.com:" in repo_info:  # SSH URL format
            path = repo_info.split("github.com:")[1].strip()
        else:  # HTTPS URL format
            path = repo_info.split("github.com/")[1].strip()

        if path.endswith(".git"):
            path = path[:-4]

        github_pages_url = (
            f"https://{path.split('/')[0]}.github.io/{path.split('/')[1]}"
        )
    else:
        github_pages_url = "https://vinhnx.github.io/VT.ai"

    print(f"\nGitHub Pages URL: {github_pages_url}")
    print(
        "Note: It may take a few minutes for GitHub Pages to update after deployment."
    )
    print("Checking for llms.txt at the GitHub Pages URL...")

    # Check if curl is available
    if check_command("which curl"):
        print("\nWaiting 20 seconds for GitHub Pages to update...")
        time.sleep(20)  # Give GitHub Pages some time to update

        # Try to access the llms.txt file
        for i in range(3):  # Try up to 3 times
            status = check_command(
                f"curl -s -o /dev/null -w '%{{http_code}}' {github_pages_url}/llms.txt"
            )
            if status == "200":
                print(f"✅ llms.txt is accessible at {github_pages_url}/llms.txt")
                return True
            elif i < 2:  # Don't wait after the last attempt
                print(
                    f"⚠️  llms.txt is not yet accessible (HTTP status: {status}). Waiting another 10 seconds..."
                )
                time.sleep(10)

        print(f"⚠️  llms.txt is not accessible at {github_pages_url}/llms.txt")
        print("This might be due to:")
        print("1. GitHub Pages cache not updated yet (can take up to 10 minutes)")
        print("2. The file not being properly deployed")
        print("3. GitHub Pages configuration issues")

        print("\nSuggestions:")
        print("1. Wait a few minutes and check again manually")
        print("2. Verify GitHub Pages is enabled in your repository settings")
        print("3. Ensure your gh-pages branch contains the llms.txt file")
        print("4. Try executing scripts/deploy_llms_txt.py directly")
    else:
        print("curl not available. Cannot verify deployment.")
        print(
            f"Please check manually if llms.txt is accessible at {github_pages_url}/llms.txt after a few minutes."
        )

    return False


def main():
    # 1. Get current version and ask for new version
    current_version = get_current_version()
    print(f"Current version: {current_version}")
    new_version = input("Enter new version number (or press Enter to keep current): ")

    if not new_version:
        new_version = current_version
        print(f"Keeping version at {current_version}")
    else:
        # Update version in pyproject.toml
        update_version(new_version)

    # 2. Create git tag (optional)
    tag_name = f"v{new_version}"
    create_tag = input(f"Create git tag '{tag_name}'? (y/n): ").lower()

    tag_created = False
    if create_tag == "y":
        if tag_exists(tag_name):
            print(f"⚠️  Tag '{tag_name}' already exists, skipping tag creation")
        else:
            run_command(
                f'git tag -a {tag_name} -m "Version {new_version}"',
                f"Creating git tag {tag_name}",
            )
            print(f"✅ Git tag '{tag_name}' created successfully")
            tag_created = True

    # 3. Build distribution packages
    run_command(
        "uv pip install --upgrade build twine",
        "Installing/upgrading build and twine",
    )
    run_command("rm -rf dist/*", "Cleaning dist directory")
    run_command("python -m build", "Building distribution packages")

    # 4. Upload to PyPI
    upload = input("Upload to PyPI? (y/n): ").lower()
    if upload == "y":
        run_command("python -m twine upload dist/*", "Uploading to PyPI")

    # 5. Build and deploy documentation (optional)
    update_docs = input("Build and deploy documentation? (y/n): ").lower()
    if update_docs == "y":
        # Build documentation
        docs_built = build_docs()
        if docs_built:
            print("✅ Documentation built successfully")

            # Ask if user wants to deploy docs
            deploy_docs_choice = input(
                "Deploy documentation to GitHub Pages? (y/n): "
            ).lower()
            if deploy_docs_choice == "y":
                if deploy_docs():
                    print("✅ Documentation successfully deployed to GitHub Pages")
                else:
                    print("⚠️  Documentation deployment failed")

                # Special option for llms.txt only
                if not Path("site/llms.txt").exists():
                    deploy_llms_only = input(
                        "Deploy just llms.txt file separately? (y/n): "
                    ).lower()
                    if deploy_llms_only == "y":
                        run_command(
                            "python scripts/deploy_llms_txt.py --verify --force-build",
                            "Deploying only llms.txt file",
                        )
            else:
                print("Documentation built but not deployed.")
        else:
            print("⚠️  Documentation build failed")

    # 6. Push tags and changes to GitHub
    push_to_github = input("Push tags and changes to GitHub? (y/n): ").lower()
    if push_to_github == "y":
        changes_to_push = False
        tags_to_push = False

        # Check if there are any changes to push
        status = run_command("git status --porcelain", "Checking git status")
        if status:
            # Use the new version number as the default commit message
            default_commit_msg = f"Version {new_version}"
            commit_msg = input(
                f"Enter commit message (press Enter to use '{default_commit_msg}'): "
            )
            if not commit_msg:
                commit_msg = default_commit_msg

            run_command("git add .", "Staging all changes")
            run_command(f'git commit -m "{commit_msg}"', "Committing changes")
            changes_to_push = True

        # Check if we have a tag to push
        if create_tag == "y" or tag_exists(tag_name):
            tags_to_push = True

        # Push changes if there are any
        if changes_to_push:
            run_command("git push", "Pushing commits to GitHub")

        # Push tags if there are any
        if tags_to_push:
            # Use git push origin tagname instead of git push --tags
            # This is more reliable across different git versions and configurations
            run_command(
                f"git push origin {tag_name}", f"Pushing tag {tag_name} to GitHub"
            )

        if not changes_to_push and not tags_to_push:
            print("No changes or tags to push.")

    print("\n✅ Release process completed successfully!")


if __name__ == "__main__":
    main()
