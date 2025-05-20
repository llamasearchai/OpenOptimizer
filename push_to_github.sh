#!/bin/bash

# Script to push OpenOptimizer to GitHub
# Usage: ./push_to_github.sh <github_token> [branch_name]

if [ $# -lt 1 ]; then
    echo "Usage: $0 <github_token> [branch_name]"
    echo "Example: $0 ghp_abcdefghijklmnop main"
    exit 1
fi

GITHUB_TOKEN=$1
BRANCH=${2:-main}
REPO_URL="https://github.com/llamasearchai/OpenOptimizer.git"
AUTH_REPO_URL="https://${GITHUB_TOKEN}@github.com/llamasearchai/OpenOptimizer.git"

echo "Pushing OpenOptimizer to GitHub repository: $REPO_URL"
echo "Using branch: $BRANCH"

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Error: git is not installed. Please install git first."
    exit 1
fi

# Initialize git repository if not already initialized
if [ ! -d .git ]; then
    echo "Initializing git repository..."
    git init
fi

# Check if the remote exists, if not add it
if ! git remote | grep -q "origin"; then
    echo "Adding GitHub remote..."
    git remote add origin $AUTH_REPO_URL
else
    echo "Updating remote URL..."
    git remote set-url origin $AUTH_REPO_URL
fi

# Configure Git user if not already configured
if [ -z "$(git config --get user.name)" ]; then
    git config user.name "OpenOptimizer Bot"
fi
if [ -z "$(git config --get user.email)" ]; then
    git config user.email "bot@openoptimizer.org"
fi

# Add all files
echo "Adding all files..."
git add .

# Commit changes
echo "Committing changes..."
git commit -m "Initial commit of OpenOptimizer framework"

# Create branch if not exists and checkout
if ! git show-ref --quiet --verify refs/heads/$BRANCH; then
    echo "Creating branch $BRANCH..."
    git branch $BRANCH
fi

echo "Checking out branch $BRANCH..."
git checkout $BRANCH

# Push to GitHub
echo "Pushing to GitHub..."
git push -u origin $BRANCH --force

echo "Done! OpenOptimizer has been pushed to $REPO_URL" 