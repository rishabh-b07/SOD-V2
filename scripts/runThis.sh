DATA_URL="https://example.com/drone_dataset.zip"  
DATASET_DIR="drone_data"  

if [ ! -d "$DATASET_DIR" ]; then
  mkdir -p "$DATASET_DIR"
fi

echo "Downloading drone image dataset..."
wget -q -O "$DATASET_DIR.zip" "$DATA_URL"

if [[ "$DATA_URL" =~ \.zip$ ]]; then
  echo "Extracting downloaded data..."
  unzip -q "$DATASET_DIR.zip" -d "$DATASET_DIR"
  rm "$DATASET_DIR.zip"
fi

echo "Data download complete!"
