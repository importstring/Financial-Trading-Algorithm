#!/bin/bash
# Script to set up daily cron job for stock data updates

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up daily stock data update cron job...${NC}"

# Get the absolute path to the project directory
PROJECT_DIR=$(cd "$(dirname "$0")/../../" && pwd)
SCRIPT_PATH="$PROJECT_DIR/Data/DataManagement/daily_update.py"

# Ensure the script is executable
chmod +x "$SCRIPT_PATH"

# Check if EODHD_API_KEY is set
if [ -z "$EODHD_API_KEY" ]; then
    echo -e "${YELLOW}Warning: EODHD_API_KEY environment variable is not set.${NC}"
    echo -e "You can either:"
    echo -e "  1. Export it in your shell profile: export EODHD_API_KEY=your_key"
    echo -e "  2. Add it directly to the cron job (less secure)"
    echo ""
    read -p "Enter your EODHD API key (or press enter to skip): " API_KEY
fi

# Create the cron job command
if [ -n "$API_KEY" ]; then
    # Use the provided API key directly in the cron job
    CRON_CMD="30 16 * * 1-5 cd $PROJECT_DIR && EODHD_API_KEY=$API_KEY python3 $SCRIPT_PATH > $PROJECT_DIR/data/update_logs/cron_update.log 2>&1"
else
    # Rely on the environment variable being set in the user's profile
    CRON_CMD="30 16 * * 1-5 cd $PROJECT_DIR && python3 $SCRIPT_PATH > $PROJECT_DIR/data/update_logs/cron_update.log 2>&1"
fi

# Create temporary file with existing crontab content
crontab -l > /tmp/current_cron 2>/dev/null || echo "" > /tmp/current_cron

# Check if the job already exists
if grep -q "$SCRIPT_PATH" /tmp/current_cron; then
    echo -e "${YELLOW}A cron job for stock data updates already exists. Updating it...${NC}"
    # Remove existing line with this script
    sed -i.bak "/$(echo $SCRIPT_PATH | sed 's/\//\\\//g')/d" /tmp/current_cron
fi

# Add new cron job
echo "$CRON_CMD" >> /tmp/current_cron

# Install new crontab
if crontab /tmp/current_cron; then
    echo -e "${GREEN}Success! Cron job has been set up to run daily at 4:30 PM ET on weekdays.${NC}"
    echo -e "The job will:"
    echo -e "  - Run after market close"
    echo -e "  - Update S&P 500 data using the bulk API"
    echo -e "  - Log output to $PROJECT_DIR/data/update_logs/cron_update.log"
else
    echo -e "${RED}Failed to set up cron job. Please check permissions and try again.${NC}"
fi

# Clean up
rm /tmp/current_cron
rm -f /tmp/current_cron.bak

echo ""
echo -e "${GREEN}To manually run a data update:${NC}"
echo -e "  cd $PROJECT_DIR && python3 $SCRIPT_PATH"
echo ""
echo -e "${GREEN}To run a first-time historical data load:${NC}"
echo -e "  cd $PROJECT_DIR && python3 $SCRIPT_PATH --first-time"
echo "" 