#!/bin/bash

# Auto-save script for Anki Flashcard Generator
# This script automatically commits and pushes all changes to GitHub

echo "🔄 Auto-saving to GitHub..."

# Add all changes
git add .

# Commit with timestamp
git commit -m "Auto-save: $(date '+%Y-%m-%d %H:%M:%S')"

# Push to GitHub
git push

echo "✅ Successfully saved to GitHub!"
echo "📅 Timestamp: $(date '+%Y-%m-%d %H:%M:%S')" 