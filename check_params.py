"""
Run with:
python check_params.py --config training/train_config.yml
"""

import yaml
import argparse
from renderformer.models.config import RenderFormerConfig
from renderformer.models.renderformer import RenderFormer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="training/train_config.yml", help="Path to the training config file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    model_config_dict = config_dict.get('model', {})
    config = RenderFormerConfig(**model_config_dict)
    
    model = RenderFormer(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    transformer_params = sum(p.numel() for p in model.transformer.parameters()) if hasattr(model, 'transformer') else 0
    view_transformer_params = sum(p.numel() for p in model.view_transformer.parameters()) if hasattr(model, 'view_transformer') else 0

    out_dpt_params = sum(p.numel() for p in model.view_transformer.out_dpt.parameters()) if hasattr(model.view_transformer, 'out_dpt') else 0
    
    print(f"Model Configuration: {args.config}")
    # print(f"Model:\n", model)
    print(f"Total parameters:        {total_params:,}")
    print(f"Trainable parameters:    {trainable_params:,}")
    print(f"Transformer params:      {transformer_params:,}")
    print(f"View Transformer params: {view_transformer_params:,}")
    print(f"\tOut DPT params:          {out_dpt_params:,}")

if __name__ == '__main__':
    main()
