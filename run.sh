CONFIG="configs/infer.yaml"
CKPT="checkpoints/checkpoint.pt"
TEMPLATE="configs/physical/down_template.json"


python pipeline.py \
    --config $CONFIG \
    --checkpoint $CKPT \
    --template-config $TEMPLATE \
    --scene-name "cake" \
    --output-dir "results"