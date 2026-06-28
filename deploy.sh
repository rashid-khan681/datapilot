#!/bin/bash
# DataPilot Cloud Run Deployment Script

set -e

# ANSI colors
GREEN="\033[92m"
YELLOW="\033[93m"
RED="\033[91m"
CYAN="\033[96m"
RESET="\033[0m"

echo -e "${CYAN}============================================================${RESET}"
echo -e "${CYAN}🚀 DATAPILOT: Google Cloud Run Deployment Engine${RESET}"
echo -e "${CYAN}============================================================${RESET}"

# 1. Load environment variables for local testing configuration
if [ -f .env ]; then
    echo -e "Loading configurations from .env..."
    export $(grep -v '^#' .env | xargs)
else
    echo -e "${YELLOW}⚠ Warning: .env file not found. Ensure GOOGLE_API_KEY is configured in your environment.${RESET}"
fi

# 2. Check GCP requirements
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ Error: Google Cloud SDK (gcloud CLI) is not installed.${RESET}"
    echo -e "Please install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# 3. Retrieve configurations
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ] || [ "$PROJECT_ID" = "(unset)" ]; then
    read -p "Enter your Google Cloud Project ID: " PROJECT_ID
fi

REGION="us-east1"
read -p "Enter deployment region [default: us-east1]: " USER_REGION
if [ -not -z "$USER_REGION" ]; then
    REGION=$USER_REGION
fi

SERVICE_NAME="datapilot"
IMAGE_TAG="gcr.io/$PROJECT_ID/$SERVICE_NAME:latest"

echo -e "\nConfiguration summary:"
echo -e "• GCP Project ID:  ${GREEN}$PROJECT_ID${RESET}"
echo -e "• Deploy Region:   ${GREEN}$REGION${RESET}"
echo -e "• Service Name:    ${GREEN}$SERVICE_NAME${RESET}"
echo -e "• Container Image: ${GREEN}$IMAGE_TAG${RESET}"
echo -e "------------------------------------------------------------"

# 4. Build and Local Health Check
echo -e "\n${CYAN}[1/4] Building Docker image locally...${RESET}"
docker build -t "$SERVICE_NAME:local" .

echo -e "\n${CYAN}[2/4] Testing container health locally...${RESET}"
# Run container in background for a quick test
CONTAINER_ID=$(docker run -d -p 8000:8000 -p 7860:7860 "$SERVICE_NAME:local")

# Clean up trap on script failure or exit
trap 'echo "Cleaning up local test container..."; docker kill $CONTAINER_ID &>/dev/null || true; docker rm $CONTAINER_ID &>/dev/null || true' EXIT

echo "Waiting for services to start inside container..."
sleep 5

# Check MCP server health
echo "Calling MCP health check..."
if curl -s -f http://localhost:8000/health > /dev/null; then
    echo -e "${GREEN}✓ MCP server health check passed!${RESET}"
else
    echo -e "${RED}❌ MCP server health check failed.${RESET}"
    exit 1
fi

# Kill the test container (handled by trap)
echo -e "${GREEN}✓ Local health checks completed successfully!${RESET}"

# 5. Push Image to GCR
echo -e "\n${CYAN}[3/4] Authenticating & pushing image to Container Registry...${RESET}"
gcloud auth configure-docker --quiet
docker tag "$SERVICE_NAME:local" "$IMAGE_TAG"
docker push "$IMAGE_TAG"

# 6. Deploy to Cloud Run
echo -e "\n${CYAN}[4/4] Deploying to Google Cloud Run...${RESET}"
gcloud run deploy "$SERVICE_NAME" \
  --image "$IMAGE_TAG" \
  --platform managed \
  --region "$REGION" \
  --allow-unauthenticated \
  --port 7860 \
  --set-env-vars GOOGLE_API_KEY="$GOOGLE_API_KEY",MCP_SERVER_URL="http://localhost:8000"

# Fetch and show deployment URL
DEPLOY_URL=$(gcloud run services describe "$SERVICE_NAME" --platform managed --region "$REGION" --format 'value(status.url)')

echo -e "\n${GREEN}============================================================${RESET}"
echo -e "${GREEN}🎉 DataPilot successfully deployed to Google Cloud Run!${RESET}"
echo -e "• Deployment URL: ${CYAN}$DEPLOY_URL${RESET}"
echo -e "${GREEN}============================================================${RESET}"
