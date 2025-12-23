#!/bin/bash
set -e

echo "Baking docker images..."

docker buildx bake

echo "Bake OK. Starting Docker Compose..."

docker compose up
