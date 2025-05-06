#!/usr/bin/env python3
"""
VT.ai - Release automation script
This script handles:
1. Version bumping in pyproject.toml
2. Optional git tagging
3. Building distribution packages
4. Uploading to PyPI
5. Optional documentation building and deployment
6. Optional pushing of tags and changes to GitHub

Usage:
    - Interactive mode: python release.py
    - One-click mode: python release.py --yes [--new-version X.Y.Z]
"""

import argparse
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


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="VT.ai release automation script")
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Non-interactive mode: automatically confirm all prompts",
    )
    parser.add_argument(
        "--new-version", help="Specify new version number (format: X.Y.Z)"
    )
    parser.add_argument(
        "--skip-tag", action="store_true", help="Skip creating a git tag"
    )
    parser.add_argument(
        "--skip-upload", action="store_true", help="Skip uploading to PyPI"
    )
    parser.add_argument(
        "--skip-docs",
        action="store_true",
        help="Skip building and deploying documentation",
    )
    parser.add_argument(
        "--skip-push", action="store_true", help="Skip pushing to GitHub"
    )
    parser.add_argument(
        "--commit-msg", help="Custom commit message when pushing changes"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    non_interactive = args.yes

    # 1. Get current version and determine new version
    current_version = get_current_version()
    print(f"Current version: {current_version}")

    if args.new_version:
        new_version = args.new_version
        print(f"New version (from command line): {new_version}")
        update_version(new_version)
    elif non_interactive:
        # Bump patch version automatically in non-interactive mode
        version_parts = current_version.split(".")
        version_parts[-1] = str(int(version_parts[-1]) + 1)
        new_version = ".".join(version_parts)
        print(f"Auto-bumping to version: {new_version}")
        update_version(new_version)
    else:
        new_version = input(
            "Enter new version number (or press Enter to keep current): "
        )
        if not new_version:
            new_version = current_version
            print(f"Keeping version at {current_version}")
        else:
            update_version(new_version)

    # 2. Create git tag (optional)
    tag_name = f"v{new_version}"
    create_tag = not args.skip_tag

    if not non_interactive and not args.skip_tag:
        create_tag = input(f"Create git tag '{tag_name}'? (y/n): ").lower() == "y"

    tag_created = False
    if create_tag:
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
        "python -m pip install --upgrade build twine",
        "Installing/upgrading build and twine",
    )
    run_command("rm -rf dist/*", "Cleaning dist directory")
    run_command("python -m build", "Building distribution packages")

    # 4. Upload to PyPI
    upload_to_pypi = not args.skip_upload

    if not non_interactive and not args.skip_upload:
        upload_to_pypi = input("Upload to PyPI? (y/n): ").lower() == "y"

    if upload_to_pypi:
        run_command("python -m twine upload dist/*", "Uploading to PyPI")

    # 5. Build and deploy documentation (optional)
    update_docs = not args.skip_docs

    if not non_interactive and not args.skip_docs:
        update_docs = input("Build and deploy documentation? (y/n): ").lower() == "y"

    if update_docs:
        # Determine if we're in the scripts directory or project root
        script_dir = Path(__file__).parent

        # Build docs first
        build_docs_script = script_dir / "build_docs.sh"
        print("\nBuilding documentation...")
        run_command(f"bash {build_docs_script}", "Building documentation with MkDocs")

        # In non-interactive mode, always deploy docs if building them
        deploy_docs = non_interactive

        if not non_interactive:
            deploy_docs = (
                input("Deploy documentation to GitHub Pages? (y/n): ").lower() == "y"
            )

        if deploy_docs:
            deploy_docs_script = script_dir / "deploy_docs.sh"
            run_command(
                f"bash {deploy_docs_script}", "Deploying documentation to GitHub Pages"
            )
            print("✅ Documentation successfully deployed to GitHub Pages")
        else:
            print("Documentation built but not deployed.")

    # 6. Push tags and changes to GitHub
    push_to_github = not args.skip_push

    if not non_interactive and not args.skip_push:
        push_to_github = (
            input("Push tags and changes to GitHub? (y/n): ").lower() == "y"
        )

    if push_to_github:
        changes_to_push = False
        tags_to_push = False

        # Check if there are any changes to push
        status = run_command("git status --porcelain", "Checking git status")
        if status:
            # Use the new version number as the default commit message
            default_commit_msg = f"Version {new_version}"

            if args.commit_msg:
                commit_msg = args.commit_msg
            elif non_interactive:
                commit_msg = default_commit_msg
            else:
                commit_msg = input(
                    f"Enter commit message (press Enter to use '{default_commit_msg}'): "
                )
                if not commit_msg:
                    commit_msg = default_commit_msg

            run_command("git add .", "Staging all changes")
            run_command(f'git commit -m "{commit_msg}"', "Committing changes")
            changes_to_push = True

        # Check if we have a tag to push
        if create_tag or tag_exists(tag_name):
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
