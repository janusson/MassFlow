#!/bin/bash

# MassFlow Installation and Setup Script
# This script streamlines the environment setup for analytical chemists.

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== MassFlow Setup Wizard ===${NC}\n"

# 1. Check for uv
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}uv is not installed. Installing uv...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the environment for the current session
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo -e "${GREEN}✓ uv is already installed.${NC}"
fi

# 2. Install dependencies
echo -e "\n${YELLOW}Installing project dependencies...${NC}"
uv sync
echo -e "${GREEN}✓ Dependencies installed successfully.${NC}"

# 3. Check for ProteoWizard msconvert (Optional but recommended)
echo -e "\n${YELLOW}Checking for ProteoWizard msconvert (required for 'massflow convert')...${NC}"
if command -v msconvert &> /dev/null; then
    echo -e "${GREEN}✓ msconvert found.${NC}"
else
    echo -e "${RED}✗ msconvert NOT found.${NC}"
    echo -e "  If you need to convert vendor raw files (.raw, .d), please install ProteoWizard."
    echo -e "  Visit: https://proteowizard.sourceforge.io/download.html"
fi

# 4. Final Guidance
echo -e "\n${GREEN}=== Setup Complete! ===${NC}"
echo -e "You can now start using MassFlow."
echo -e "\n${YELLOW}Recommended next steps:${NC}"
echo -e "1. Create your project configuration using the new wizard:"
echo -e "   ${GREEN}uv run massflow config-wizard${NC}"
echo -e "2. Run the tutorial to see it in action:"
echo -e "   ${GREEN}uv run massflow tutorial${NC}"
echo -e "3. Start annotating your data:"
echo -e "   ${GREEN}uv run massflow annotate --config massflow_config.yaml${NC}"
