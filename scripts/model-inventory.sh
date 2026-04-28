#!/bin/bash
# List all MLX models on disk with size and quantization info.
# Usage: ./scripts/model-inventory.sh

MODELS_DIR="${HOME}/models"

if [ ! -d "$MODELS_DIR" ]; then
    echo "No models directory at $MODELS_DIR"
    exit 1
fi

echo "=== MLX Model Inventory ==="
echo "Directory: $MODELS_DIR"
echo ""

printf "%-50s %10s %s\n" "MODEL" "SIZE" "QUANT INFO"
printf "%-50s %10s %s\n" "-----" "----" "----------"

for model_dir in "$MODELS_DIR"/*/; do
    [ -d "$model_dir" ] || continue
    name=$(basename "$model_dir")
    size=$(du -sh "$model_dir" 2>/dev/null | cut -f1)

    # Extract quantization info from config.json if present
    config="$model_dir/config.json"
    quant_info=""
    if [ -f "$config" ]; then
        bits=$(python3 -c "import json; c=json.load(open('$config')); print(c.get('quantization',{}).get('bits','?'))" 2>/dev/null)
        group_size=$(python3 -c "import json; c=json.load(open('$config')); print(c.get('quantization',{}).get('group_size','?'))" 2>/dev/null)
        quant_type=$(python3 -c "import json; c=json.load(open('$config')); print(c.get('quantization',{}).get('quant_type','affine'))" 2>/dev/null)
        if [ "$bits" != "?" ]; then
            quant_info="${bits}-bit ${quant_type} gs=${group_size}"
        fi
    fi

    printf "%-50s %10s %s\n" "$name" "$size" "$quant_info"
done

echo ""
echo "=== HF Cache ==="
if [ -d "$HOME/.cache/huggingface/hub" ]; then
    cache_size=$(du -sh "$HOME/.cache/huggingface/hub" 2>/dev/null | cut -f1)
    echo "Size: $cache_size"
else
    echo "No HF cache found"
fi
