#!/disk/xray15/aem2/envs/camels/bin/python
# # Script to run hyperparameter search for SBI training

import os
import sys
import argparse
import json
import itertools
from datetime import datetime
from sbi_training_LH import train_sbi_model

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run SBI training with specified hyperparameters')
    
    # Network architecture parameters
    parser.add_argument('--hidden_features', type=int, default=60,
                        help='Number of hidden features in neural network')
    parser.add_argument('--num_transforms', type=int, default=5,
                        help='Number of transforms in neural network')
    parser.add_argument('--num_nets', type=int, default=2,
                        help='Number of networks in ensemble')
    
    # Training parameters
    parser.add_argument('--training_batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='Learning rate for training')
    parser.add_argument('--stop_after_epochs', type=int, default=20,
                        help='Number of epochs without improvement before early stopping')
    parser.add_argument('--max_num_epochs', type=int, default=100,
                        help='Maximum number of epochs to train for')
    parser.add_argument('--clip_max_norm', type=float, default=1.0,
                        help='Maximum gradient norm')
    parser.add_argument('--validation_fraction', type=float, default=0.1,
                        help='Fraction of data to use for validation')
    
    # Data parameters
    parser.add_argument('--model', type=str, default='IllustrisTNG',
                        help='Simulation model to use')
    parser.add_argument('--spec_type', type=str, default='attenuated',
                        help='Spectral type to use')
    parser.add_argument('--sps', type=str, default='BC03',
                        help='Stellar population synthesis model to use')
    parser.add_argument('--snap', type=str, default='044',
                        help='Snapshot to use')
    parser.add_argument('--bands', type=str, default='all',
                        help='Bands to use')
    parser.add_argument('--no_colours', action='store_false', dest='colours',
                        help='Disable use of colours')
    parser.add_argument('--no_luminosity_functions', action='store_false', 
                        dest='luminosity_functions', help='Disable use of luminosity functions')
    
    # Grid search parameters
    parser.add_argument('--grid_search', action='store_true',
                        help='Run grid search over specified hyperparameters')
    parser.add_argument('--grid_config', type=str, default=None,
                        help='Path to JSON file specifying grid search parameters')
    
    # Set defaults for boolean arguments
    parser.set_defaults(colours=True, luminosity_functions=True)
    
    return parser.parse_args()

def run_single_training(args):
    """Run a single training with the specified arguments."""
    print("\n" + "="*80)
    print(f"Running training with hyperparameters:")
    print("-"*80)
    for arg, value in vars(args).items():
        if arg not in ['grid_search', 'grid_config']:
            print(f"{arg}: {value}")
    print("="*80 + "\n")
    
    # Convert snap to list format
    snap = [args.snap]
    
    # Run training
    results = train_sbi_model(
        hidden_features=args.hidden_features,
        num_transforms=args.num_transforms,
        num_nets=args.num_nets,
        training_batch_size=args.training_batch_size,
        learning_rate=args.learning_rate,
        stop_after_epochs=args.stop_after_epochs,
        max_num_epochs=args.max_num_epochs,
        clip_max_norm=args.clip_max_norm,
        validation_fraction=args.validation_fraction,
        model=args.model,
        spec_type=args.spec_type,
        sps=args.sps,
        snap=snap,
        bands=args.bands,
        colours=args.colours,
        luminosity_functions=args.luminosity_functions
    )
    
    return results

def run_grid_search(args, config_file):
    """Run a grid search over hyperparameters specified in config file."""
    # Load grid search parameters from config file
    with open(config_file, 'r') as f:
        grid_params = json.load(f)
    
    print(f"Loaded grid search parameters from {config_file}")
    print("Grid search will explore these parameter combinations:")
    for param, values in grid_params.items():
        print(f"  {param}: {values}")
    
    # Create all combinations of parameters
    param_names = list(grid_params.keys())
    param_values = list(grid_params.values())
    combinations = list(itertools.product(*param_values))
    
    print(f"\nTotal number of combinations to explore: {len(combinations)}")
    
    # Create a log file for grid search results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"grid_search_results_{timestamp}.txt"
    
    with open(log_file, 'w') as f:
        f.write(f"Grid search started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total combinations: {len(combinations)}\n\n")
        f.write("Parameter grid:\n")
        for param, values in grid_params.items():
            f.write(f"  {param}: {values}\n")
        f.write("\n" + "="*80 + "\n\n")
    
    # Run training for each combination
    for i, combo in enumerate(combinations):
        print(f"\nRunning combination {i+1}/{len(combinations)}")
        
        # Create a copy of args and update with current combination values
        current_args = argparse.Namespace(**vars(args))
        for param_name, param_value in zip(param_names, combo):
            setattr(current_args, param_name, param_value)
        
        # Log current combination
        with open(log_file, 'a') as f:
            f.write(f"Combination {i+1}/{len(combinations)}:\n")
            for param_name, param_value in zip(param_names, combo):
                f.write(f"  {param_name}: {param_value}\n")
        
        try:
            # Run training with current combination
            results = run_single_training(current_args)
            
            # Log success
            with open(log_file, 'a') as f:
                f.write("  Status: SUCCESS\n")
                
                # Add some metrics if available (this would need to be extended)
                if 'summaries' in results and results['summaries']:
                    validation_losses = results['summaries'][0].get('validation_log_probs', [])
                    if validation_losses:
                        best_loss = max(validation_losses)
                        best_epoch = validation_losses.index(best_loss)
                        f.write(f"  Best validation loss: {best_loss:.6f} (epoch {best_epoch})\n")
                
        except Exception as e:
            # Log failure
            with open(log_file, 'a') as f:
                f.write(f"  Status: FAILED\n")
                f.write(f"  Error: {str(e)}\n")
            
            print(f"Error in combination {i+1}: {str(e)}")
            
        # Add separator
        with open(log_file, 'a') as f:
            f.write("\n" + "-"*80 + "\n\n")
    
    # Log completion
    with open(log_file, 'a') as f:
        f.write(f"\nGrid search completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"\nGrid search completed. Results logged to: {log_file}")

def main():
    """Main function to run training or grid search."""
    args = parse_args()
    
    if args.grid_search:
        if args.grid_config is None:
            print("Error: Grid search requires a config file. Use --grid_config to specify.")
            sys.exit(1)
        
        if not os.path.exists(args.grid_config):
            print(f"Error: Grid config file {args.grid_config} not found.")
            sys.exit(1)
        
        run_grid_search(args, args.grid_config)
    else:
        run_single_training(args)

if __name__ == "__main__":
    main()