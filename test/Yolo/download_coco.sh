#!/usr/bin/env bash

# Simple helper to download MS-COCO 2017 into data/coco/
# Run from the project root or from examples/ (it uses relative paths).

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data/coco"

mkdir -p "${DATA_DIR}/images"
mkdir -p "${DATA_DIR}/annotations"

echo "Downloading COCO 2017 train/val images and annotations into ${DATA_DIR} ..."

cd "${DATA_DIR}"

# Train images
if [ ! -f "train2017.zip" ]; then
  echo "Downloading train2017..."
  wget http://images.cocodataset.org/zips/train2017.zip
fi

# Val images
if [ ! -f "val2017.zip" ]; then
  echo "Downloading val2017..."
  wget http://images.cocodataset.org/zips/val2017.zip
fi

# Annotations
if [ ! -f "annotations_trainval2017.zip" ]; then
  echo "Downloading annotations..."
  wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
fi

echo "Extracting archives..."

unzip -n train2017.zip -d images
unzip -n val2017.zip -d images
unzip -n annotations_trainval2017.zip -d .

echo "Done. Expected structure:"
echo "  data/coco/images/train2017/"
echo "  data/coco/images/val2017/"
echo "  data/coco/annotations/instances_train2017.json"
echo "  data/coco/annotations/instances_val2017.json"
