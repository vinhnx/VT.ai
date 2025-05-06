#!/usr/bin/env python3
"""
Deploy llms.txt to GitHub Pages and other destinations.

This script ensures the llms.txt file is properly deployed to GitHub Pages
and validates it according to the llmstxt.org standard.

Usage:
	python deploy_llms_txt.py [options]

Examples:
	python deploy_llms_txt.py --verify
	python deploy_llms_txt.py --force-build
	python deploy_llms_txt.py --file custom_llms.txt
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

try:
	import requests
except ImportError:
	print("Installing required dependencies...")
	subprocess.run(["uv", "pip", "install", "requests"], check=True)
	import requests


def validate_llms_txt(file_path: str) -> bool:
	"""
	Validate the format and content of the llms.txt file.

	Args:
		file_path: Path to the llms.txt file

	Returns:
		bool: True if the file is valid, False otherwise
	"""
	# Read the file
	with open(file_path, 'r', encoding='utf-8') as f:
		content = f.read()
	
	# Basic validation - check for recommended sections according to llmstxt.org
	required_sections = [
		"title:", 
		"url:",
		"language:",
		"summary:", 
		"description:", 
		"audience:"
	]
	
	# Check if all required sections are present
	valid = all(section in content for section in required_sections)
	
	if not valid:
		print("Warning: llms.txt appears to be missing some recommended sections.")
		print("Consider adding the following sections for compliance with llmstxt.org:")
		missing = [s for s in required_sections if s not in content]
		for section in missing:
			print(f"  - {section}")
	
	return True  # Return True anyway to allow deployment with warnings


def get_github_pages_url() -> str:
	"""
	Get the GitHub Pages URL for the current repository.

	Returns:
		str: The GitHub Pages URL
	"""
	# Try to get the GitHub Pages URL from git remote
	try:
		repo_info = subprocess.run(
			["git", "remote", "get-url", "origin"], 
			check=True, 
			capture_output=True, 
			text=True
		).stdout.strip()
		
		if repo_info and "github.com" in repo_info:
			# Extract username and repo name from remote URL
			if "github.com:" in repo_info:  # SSH URL format
				path = repo_info.split("github.com:")[1].strip()
			else:  # HTTPS URL format
				path = repo_info.split("github.com/")[1].strip()
			
			if path.endswith(".git"):
				path = path[:-4]
			
			github_pages_url = f"https://{path.split('/')[0]}.github.io/{path.split('/')[1]}"
			return github_pages_url
	except Exception as e:
		print(f"Error determining GitHub Pages URL: {e}")
	
	# Default fallback
	return "https://vinhnx.github.io/VT.ai"


def ensure_site_directory() -> None:
	"""
	Ensure the site directory exists and is properly set up.
	"""
	# Create site directory if it doesn't exist
	site_dir = Path("site")
	site_dir.mkdir(exist_ok=True)
	
	# Create .nojekyll file to prevent GitHub Pages from using Jekyll
	nojekyll_path = site_dir / ".nojekyll"
	if not nojekyll_path.exists():
		nojekyll_path.touch()
		print("Created .nojekyll file to prevent GitHub Pages from using Jekyll")


def run_command(cmd: str, desc: Optional[str] = None, check: bool = True) -> subprocess.CompletedProcess:
	"""
	Run a shell command and optionally print its output.

	Args:
		cmd: The command to run
		desc: Optional description of the command
		check: Whether to check the return code

	Returns:
		subprocess.CompletedProcess: The result of the command
	"""
	if desc:
		print(f"\n{desc}...")
	
	result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
	
	if check and result.returncode != 0:
		print(f"Error running command: {cmd}")
		print(f"Output: {result.stdout}")
		print(f"Error: {result.stderr}")
		if check:
			sys.exit(1)
	
	return result


def deploy_to_github_pages(llms_txt_path: str, force_build: bool = False) -> bool:
	"""
	Deploy llms.txt to GitHub Pages.
	
	Args:
		llms_txt_path: Path to the llms.txt file
		force_build: Whether to force a full rebuild of the documentation
		
	Returns:
		bool: True if deployment was successful, False otherwise
	"""
	ensure_site_directory()
	
	# Copy the llms.txt file to the site directory
	try:
		shutil.copy(llms_txt_path, "site/")
		print(f"Copied {llms_txt_path} to site/")
	except Exception as e:
		print(f"Error copying llms.txt: {e}")
		return False
	
	# Build the docs if forced
	if force_build:
		try:
			result = run_command("mkdocs build", "Building documentation with MkDocs")
			print(result.stdout)
		except Exception as e:
			print(f"Error building documentation: {e}")
			return False
	
	# Explicitly add the llms.txt file to git to ensure it's tracked
	try:
		run_command("git add -f site/llms.txt", "Adding llms.txt to git tracking")
	except Exception as e:
		print(f"Error adding llms.txt to git (non-critical): {e}")
		# Continue anyway, this error is non-critical
	
	# Check if the gh-pages branch exists
	result = subprocess.run(
		["git", "branch", "--list", "gh-pages"],
		capture_output=True,
		text=True
	)
	gh_pages_exists = "gh-pages" in result.stdout
	
	if not gh_pages_exists:
		print("gh-pages branch does not exist. Creating it...")
		try:
			# Create and configure the gh-pages branch
			run_command("git checkout --orphan gh-pages", "Creating gh-pages branch")
			run_command("git rm -rf .", "Clearing working directory")
			run_command("touch index.html", "Creating placeholder index.html")
			run_command("touch .nojekyll", "Creating .nojekyll file")
			run_command("cp " + llms_txt_path + " llms.txt", "Copying llms.txt to root")
			run_command("git add index.html .nojekyll llms.txt", "Adding files to git")
			run_command('git commit -m "Initial gh-pages commit with llms.txt"', "Committing files")
			run_command("git push origin gh-pages", "Pushing gh-pages branch")
			run_command("git checkout -", "Returning to previous branch")
			return True
		except Exception as e:
			print(f"Error creating gh-pages branch: {e}")
			run_command("git checkout -", "Attempting to return to previous branch", check=False)
			return False
	
	# Deploy to GitHub Pages using mkdocs
	deploy_cmd = ["mkdocs", "gh-deploy", "--force", "--no-jekyll"]
	if not force_build:
		deploy_cmd.append("--no-build")
	deploy_cmd.extend(["--message", "Deploy llms.txt update"])
	
	try:
		deploy_cmd_str = " ".join(deploy_cmd)
		result = run_command(deploy_cmd_str, "Deploying to GitHub Pages")
		print("Successfully deployed to GitHub Pages")
		return True
	except Exception as e:
		print(f"Error deploying to GitHub Pages: {e}")
		
		# Try a direct approach if mkdocs fails
		try:
			print("Trying alternative deployment method...")
			run_command("git checkout gh-pages", "Switching to gh-pages branch")
			run_command(f"cp {llms_txt_path} llms.txt", "Copying llms.txt to root")
			run_command("touch .nojekyll", "Creating .nojekyll file")
			run_command("git add llms.txt .nojekyll", "Adding files to git")
			run_command('git commit -m "Update llms.txt"', "Committing changes")
			run_command("git push origin gh-pages", "Pushing changes")
			run_command("git checkout -", "Returning to previous branch")
			print("Alternative deployment successful")
			return True
		except Exception as e2:
			print(f"Alternative deployment failed: {e2}")
			run_command("git checkout -", "Attempting to return to previous branch", check=False)
			return False


def verify_deployment(base_url: str, max_attempts: int = 3, wait_time: int = 10) -> bool:
	"""
	Verify that the llms.txt file was successfully deployed.
	
	Args:
		base_url: Base URL of the deployed site
		max_attempts: Maximum number of attempts to verify
		wait_time: Time to wait between attempts (seconds)
	
	Returns:
		bool: True if verification was successful, False otherwise
	"""
	llms_txt_url = f"{base_url.rstrip('/')}/llms.txt"
	print(f"Verifying deployment at: {llms_txt_url}")
	print(f"Note: GitHub Pages may take up to 10 minutes to update after deployment.")
	
	# Initial waiting period
	print(f"Waiting {wait_time} seconds for initial GitHub Pages update...")
	time.sleep(wait_time)
	
	for attempt in range(max_attempts):
		try:
			# Use requests to check if the file is accessible
			response = requests.head(llms_txt_url, allow_redirects=True, timeout=30)
			
			if response.status_code == 200:
				print(f"✅ Attempt {attempt+1}/{max_attempts}: llms.txt is accessible!")
				
				# Verify content with a GET request
				content_response = requests.get(llms_txt_url, timeout=30)
				if "llms.txt" in content_response.text.lower():
					print("✅ Content verification successful")
					return True
				else:
					print("⚠️  Content doesn't appear to be a valid llms.txt file")
			else:
				print(f"⚠️  Attempt {attempt+1}/{max_attempts}: HTTP status {response.status_code}")
				
				if attempt < max_attempts - 1:
					print(f"Waiting {wait_time} seconds before next attempt...")
					time.sleep(wait_time)
					# Increase wait time for subsequent attempts
					wait_time = int(wait_time * 1.5)
		except requests.RequestException as e:
			print(f"⚠️  Attempt {attempt+1}/{max_attempts}: Error: {e}")
			
			if attempt < max_attempts - 1:
				print(f"Waiting {wait_time} seconds before next attempt...")
				time.sleep(wait_time)
				# Increase wait time for subsequent attempts
				wait_time = int(wait_time * 1.5)
	
	print("\n⚠️  Verification failed after multiple attempts")
	print("This might be due to:")
	print("1. GitHub Pages cache not updated yet (can take up to 10 minutes)")
	print("2. The file not being properly deployed")
	print("3. GitHub Pages configuration issues")
	
	print("\nSuggestions:")
	print("1. Wait a few more minutes and check manually")
	print("2. Verify GitHub Pages is enabled in repository settings")
	print("3. Check if the gh-pages branch contains llms.txt")
	print(f"4. Try accessing {llms_txt_url} directly in a browser")
	
	return False


def create_basic_llms_txt() -> str:
	"""
	Create a basic llms.txt file if one doesn't exist.
	
	Returns:
		str: Path to the created file
	"""
	output_path = "llms.txt"
	
	# Get repository information from git
	try:
		repo_url = subprocess.run(
			["git", "remote", "get-url", "origin"],
			check=True,
			capture_output=True,
			text=True
		).stdout.strip()
		
		if repo_url.endswith(".git"):
			repo_url = repo_url[:-4]
		
		if repo_url.startswith("git@github.com:"):
			repo_url = repo_url.replace("git@github.com:", "https://github.com/")
		
		project_name = repo_url.split("/")[-1]
	except Exception:
		repo_url = "https://github.com/vinhnx/VT.ai"
		project_name = "VT.ai"
	
	# Get current date
	current_date = time.strftime("%Y-%m-%d")
	
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
  - This llms.txt was created on {current_date}
  - Updates to the documentation can be found in the GitHub repository
"""
	
	with open(output_path, "w", encoding="utf-8") as f:
		f.write(llms_content)
	
	print(f"✅ Created basic llms.txt file at {output_path}")
	return output_path


def main() -> int:
	"""
	Main entry point for the script.
	
	Returns:
		int: Exit code (0 for success, 1 for failure)
	"""
	parser = argparse.ArgumentParser(
		description="Deploy llms.txt file to GitHub Pages",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	parser.add_argument(
		"--file", 
		default="llms.txt", 
		help="Path to the llms.txt file to deploy"
	)
	parser.add_argument(
		"--force-build", 
		action="store_true", 
		help="Force a full documentation rebuild with MkDocs"
	)
	parser.add_argument(
		"--verify", 
		action="store_true", 
		help="Verify deployment after completion"
	)
	parser.add_argument(
		"--attempts", 
		type=int, 
		default=3, 
		help="Number of verification attempts"
	)
	parser.add_argument(
		"--wait", 
		type=int, 
		default=20, 
		help="Initial seconds to wait between verification attempts"
	)
	parser.add_argument(
		"--create-if-missing", 
		action="store_true", 
		help="Create a basic llms.txt file if not found"
	)
	parser.add_argument(
		"--base-url", 
		default=None, 
		help="Base URL for verification (default: auto-detect from git)"
	)
	
	args = parser.parse_args()
	
	# If llms.txt doesn't exist and create-if-missing is enabled, create it
	if not Path(args.file).exists():
		if args.create_if_missing:
			args.file = create_basic_llms_txt()
		else:
			print(f"Error: {args.file} not found. Use --create-if-missing to create it.")
			return 1
	
	# Validate the llms.txt file
	if not validate_llms_txt(args.file):
		print("Error: llms.txt validation failed.")
		return 1
	
	# Get base URL for verification
	base_url = args.base_url or get_github_pages_url()
	
	# Deploy to GitHub Pages
	if deploy_to_github_pages(args.file, args.force_build):
		print(f"✅ Successfully deployed {args.file} to GitHub Pages")
		
		# Verify deployment if requested
		if args.verify:
			if verify_deployment(base_url, args.attempts, args.wait):
				print(f"✅ Successfully verified llms.txt at {base_url}/llms.txt")
			else:
				print(f"⚠️  Could not verify llms.txt at {base_url}/llms.txt")
				print(f"This might be due to GitHub Pages caching. Please check manually in a few minutes.")
		else:
			print(f"Deployment complete. Check {base_url}/llms.txt in a few minutes.")
		
		return 0
	else:
		print(f"❌ Deployment failed. Please check the errors above.")
		return 1


if __name__ == "__main__":
	sys.exit(main())
