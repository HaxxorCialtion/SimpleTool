#!/bin/bash
# Run from github_test/ directory
# Reorganizes files into SimpleTool/ with numbered scripts

set -e
cd "$(dirname "$0")"

echo "=== Reorganizing SimpleTool project ==="

# 1. Create prompts/ and models/ dirs
mkdir -p SimpleTool/prompts
mkdir -p SimpleTool/models

# 2. Move prompt files into SimpleTool/prompts/
mv v1_system.txt      SimpleTool/prompts/
mv scenarios.json      SimpleTool/prompts/
mv tools_game.jsonl    SimpleTool/prompts/
mv tools_arm.jsonl     SimpleTool/prompts/
mv tools_avatar.jsonl  SimpleTool/prompts/

# 3. Rename and move scripts with numbered prefixes
mv run.py              SimpleTool/01_benchmark.py
mv SimpleTool/rt_server.py    SimpleTool/02_server.py
mv SimpleTool/test_server.py  SimpleTool/03_test_server.py

# 4. Clean up stale file if exists
rm -f test_parallel_3domains_vllm.py

echo ""
echo "=== Done! New structure ==="
find SimpleTool/ -maxdepth 2 -not -path '*/assets/*' -not -path '*/demos/neon_arena/*' | sort
