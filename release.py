#!/usr/bin/env python3
"""
VT.ai - Release automation script
This script handles:
1. Version bumping in pyproject.toml
2. Building distribution packages
3. Uploading to PyPI
4. Creating a git tag for the release
5. Pushing changes to GitHub
"""

import re
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

    # 2. Build distribution packages
    run_command(
        "python -m pip install --upgrade build twine",
        "Installing/upgrading build and twine",
    )
    run_command("rm -rf dist/*", "Cleaning dist directory")
    run_command("python -m build", "Building distribution packages")

    # 3. Upload to PyPI
    upload = input("Upload to PyPI? (y/n): ").lower()
    if upload == "y":
        run_command("python -m twine upload dist/*", "Uploading to PyPI")

    # 4. Create git commit and tag
    run_command("git add pyproject.toml", "Staging pyproject.toml changes")
    run_command(
        f'git commit -m "Bump version to {new_version}"', "Committing version change"
    )
    run_command(
        f'git tag -a v{new_version} -m "Version {new_version}"', "Creating git tag"
    )

    # 5. Push changes and tags to GitHub
    push = input("Push changes to GitHub? (y/n): ").lower()
    if push == "y":
        run_command("git push", "Pushing commits")
        run_command("git push --tags", "Pushing tags")

    print("\n✅ Release process completed successfully!")


if __name__ == "__main__":
    main()
    # Removed duplicate main() call
