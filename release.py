#!/usr/bin/env python3
"""
VT.ai - Release automation script
This script handles:
1. Version bumping in pyproject.toml
2. Building distribution packages
3. Uploading to PyPI
4. Generating a changelog from git commits
5. Creating a git tag for the release
6. Pushing changes to GitHub
7. Creating a GitHub release with changelog
"""

import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime
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


def get_commit_history(previous_tag=None, current_tag=None):
    """Get commit history since the last tag or between specified tags"""
    if not previous_tag:
        # Get the most recent tag if not specified
        try:
            previous_tag = run_command(
                "git describe --tags --abbrev=0", "Finding the most recent tag"
            )
        except:
            # If no tags exist yet, get all commits
            print("No previous tags found. Including all commits in changelog.")
            return run_command(
                "git log --pretty=format:'%h %s (%an)' --no-merges",
                "Getting all commit history",
            )

    range_spec = f"{previous_tag}..HEAD"
    if current_tag:
        range_spec = f"{previous_tag}..{current_tag}"

    return run_command(
        f"git log {range_spec} --pretty=format:'%h %s (%an)' --no-merges",
        f"Getting commit history from {range_spec}",
    )


def generate_changelog(new_version, previous_tag=None):
    """Generate a changelog from git commit history"""
    print("\nGenerating changelog...")

    changelog_path = Path("CHANGELOG.md")
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Get commit history
    commit_history = get_commit_history(previous_tag)
    if not commit_history:
        print("No new commits found for changelog.")
        return changelog_path

    # Create categorized commits if possible
    feature_commits = []
    fix_commits = []
    other_commits = []

    for line in commit_history.split("\n"):
        if not line.strip():
            continue
        if re.search(r"(feat|feature|add)[\(\):]", line.lower()):
            feature_commits.append(line)
        elif re.search(r"(fix|bug|patch)[\(\):]", line.lower()):
            fix_commits.append(line)
        else:
            other_commits.append(line)

    # Generate changelog content
    changelog_content = f"## v{new_version} ({current_date})\n\n"

    if feature_commits:
        changelog_content += "### Features\n\n"
        for commit in feature_commits:
            changelog_content += f"- {commit}\n"
        changelog_content += "\n"

    if fix_commits:
        changelog_content += "### Bug Fixes\n\n"
        for commit in fix_commits:
            changelog_content += f"- {commit}\n"
        changelog_content += "\n"

    if other_commits:
        changelog_content += "### Other Changes\n\n"
        for commit in other_commits:
            changelog_content += f"- {commit}\n"
        changelog_content += "\n"

    # Update or create the changelog file
    if changelog_path.exists():
        existing_content = changelog_path.read_text()
        changelog_path.write_text(changelog_content + existing_content)
    else:
        changelog_path.write_text(f"# Changelog\n\n{changelog_content}")

    print(f"Changelog updated at {changelog_path}")
    return changelog_path


def get_git_remote():
    """Get the default git remote (usually 'origin')"""
    try:
        remotes = run_command("git remote", "Listing git remotes")
        if not remotes:
            print("No git remotes found. Using 'origin' as default.")
            return "origin"

        # If multiple remotes, prefer 'origin'
        if "origin" in remotes.split("\n"):
            return "origin"
        # Otherwise use the first remote
        return remotes.split("\n")[0]
    except:
        print("Error getting git remotes. Using 'origin' as default.")
        return "origin"


def extract_github_repo_info():
    """Extract GitHub repository owner and name from remote URL"""
    try:
        # Get the remote URL
        remote = get_git_remote()
        remote_url = run_command(f"git remote get-url {remote}", "Getting remote URL")

        # Parse GitHub URL to extract owner and repo
        # Handle different URL formats (HTTPS or SSH)
        if "github.com" in remote_url:
            if remote_url.startswith("https://"):
                # Format: https://github.com/owner/repo.git
                parts = remote_url.strip().split("/")
                owner = parts[-2]
                repo = parts[-1]
            else:
                # Format: git@github.com:owner/repo.git
                parts = remote_url.split(":")[-1].strip().split("/")
                owner = parts[0]
                repo = parts[1]

            # Remove .git extension if present
            if repo.endswith(".git"):
                repo = repo[:-4]

            return owner, repo
    except Exception as e:
        print(f"Error extracting GitHub repository info: {e}")
        return None, None


def create_github_release(version, changelog_content):
    """Create a GitHub release with the changelog content"""
    try:
        # Extract GitHub repository info
        owner, repo = extract_github_repo_info()
        if not owner or not repo:
            print("Could not determine GitHub repository information.")
            return False

        # Check for GitHub token
        github_token = os.environ.get("GITHUB_TOKEN")
        if not github_token:
            github_token = input(
                "Enter GitHub token (or set GITHUB_TOKEN environment variable): "
            ).strip()
            if not github_token:
                print("GitHub token is required to create a release.")
                return False

        # Prepare release data
        tag_name = f"v{version}"
        release_name = f"Version {version}"

        # Use the changelog content as the release body
        # Read the latest section from the changelog file
        changelog_path = Path("CHANGELOG.md")
        if changelog_path.exists():
            changelog_text = changelog_path.read_text()
            # Extract the section for the current version
            version_pattern = f"## v{version} "
            if version_pattern in changelog_text:
                start_idx = changelog_text.index(version_pattern)
                next_section = changelog_text.find("## v", start_idx + 1)
                if next_section > 0:
                    release_body = changelog_text[start_idx:next_section].strip()
                else:
                    release_body = changelog_text[start_idx:].strip()
            else:
                # If we can't find the specific version, use the first section
                first_section_end = changelog_text.find(
                    "## v", changelog_text.find("## v") + 1
                )
                if first_section_end > 0:
                    release_body = changelog_text[:first_section_end].strip()
                else:
                    release_body = changelog_text.strip()
        else:
            # If no changelog file, use a simple message
            release_body = f"Release version {version}"

        print(f"\nCreating GitHub release for tag {tag_name}...")

        # Create the release using GitHub API
        url = f"https://api.github.com/repos/{owner}/{repo}/releases"
        headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json",
        }

        data = {
            "tag_name": tag_name,
            "name": release_name,
            "body": release_body,
            "draft": False,
            "prerelease": False,
        }

        request = urllib.request.Request(
            url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST"
        )

        with urllib.request.urlopen(request) as response:
            if response.status == 201:
                response_data = json.loads(response.read().decode("utf-8"))
                print(
                    f"✅ GitHub release created successfully: {response_data.get('html_url')}"
                )
                return True
            else:
                print(
                    f"Error creating GitHub release: {response.status} {response.reason}"
                )
                return False

    except Exception as e:
        print(f"Error creating GitHub release: {e}")
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

    # 2. Generate changelog
    try:
        # Try to get the last tag for changelog generation
        last_tag = run_command(
            "git describe --tags --abbrev=0", "Getting last tag for changelog"
        )
    except:
        last_tag = None

    changelog_path = generate_changelog(new_version, last_tag)

    # 3. Build distribution packages
    run_command(
        "python -m pip install --upgrade build twine",
        "Installing/upgrading build and twine",
    )
    run_command("rm -rf dist/*", "Cleaning dist directory")
    run_command("python -m build", "Building distribution packages")

    # 4. Upload to PyPI
    upload = input("Upload to PyPI? (y/n): ").lower()
    if upload == "y":
        run_command("python -m twine upload dist/*", "Uploading to PyPI")

    # 5. Create git commit with version and changelog
    run_command("git add pyproject.toml", "Staging pyproject.toml changes")
    run_command(f"git add {changelog_path}", "Staging changelog changes")
    run_command(
        f'git commit -m "Bump version to {new_version}"', "Committing version change"
    )
    run_command(
        f'git tag -a v{new_version} -m "Version {new_version}"', "Creating git tag"
    )

    # 6. Push changes and tags to GitHub
    push = input("Push changes to GitHub? (y/n): ").lower()
    if push == "y":
        # Get default remote
        remote = get_git_remote()
        run_command(f"git push {remote}", "Pushing commits")
        run_command(f"git push {remote} --tags", "Pushing tags")

    # 7. Create GitHub release
    create_release = input("Create GitHub release? (y/n): ").lower()
    if create_release == "y":
        changelog_content = (
            changelog_path.read_text() if changelog_path.exists() else ""
        )
        create_github_release(new_version, changelog_content)

    print("\n✅ Release process completed successfully!")


if __name__ == "__main__":
    main()
