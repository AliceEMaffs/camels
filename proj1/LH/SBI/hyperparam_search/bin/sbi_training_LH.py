#!/disk/xray15/aem2/envs/camels/bin/python
# # SBI Training Script with configurable hyperparameters

import pickle
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import torch
import seaborn as sns
from sklearn.preprocessing import Normalizer
import joblib
import ili
from ili.dataloaders import NumpyLoader
from ili.inference import InferenceRunner
from ili.validation.metrics import PosteriorCoverage, PlotSinglePosterior, PosteriorSamples

sys.path.append("/disk/xray15/aem2/camels/proj1")
from setup_params_LH import plot_uvlf, plot_colour
from setup_params_LH import *
from priors import initialise_priors
from variables_config import n_bins_lf, n_bins_colour

def train_sbi_model(hidden_features=60, num_transforms=5, num_nets=2, 
                    training_batch_size=4, learning_rate=5e-4, stop_after_epochs=20,
                    max_num_epochs=100, clip_max_norm=1, validation_fraction=0.1,
                    use_combined_loss=True, show_train_summary=True,
                    num_workers=0, pin_memory=False,
                    model="IllustrisTNG", spec_type="attenuated", sps="BC03", 
                    snap=['044'], bands="all", colours=True, luminosity_functions=True,
                    run_analysis=True):
    """
    Train an SBI model with the specified hyperparameters.
    
    Parameters:
    -----------
    hidden_features : int
        Number of hidden features in the neural network
    num_transforms : int
        Number of transforms in the neural network
    num_nets : int
        Number of networks in the ensemble
    training_batch_size : int
        Batch size for training
    learning_rate : float
        Learning rate for training
    stop_after_epochs : int
        Number of epochs without improvement before early stopping
    max_num_epochs : int
        Maximum number of epochs to train for
    clip_max_norm : float
        Maximum gradient norm
    validation_fraction : float
        Fraction of data to use for validation
    use_combined_loss : bool
        Whether to use combined loss
    show_train_summary : bool
        Whether to show training summary
    num_workers : int
        Number of workers for data loading
    pin_memory : bool
        Whether to pin memory for data loading
    model : str
        Simulation model to use
    spec_type : str
        Spectral type to use
    sps : str
        Stellar population synthesis model to use
    snap : list
        List of snapshots to use
    bands : str
        Bands to use
    colours : bool
        Whether to use colours
    luminosity_functions : bool
        Whether to use luminosity functions
    run_analysis : bool
        Whether to run analysis after training
    
    Returns:
    --------
    dict
        Dictionary containing results of training
    """
    # Create train_args dictionary from input parameters
    train_args = {
        "training_batch_size": training_batch_size,
        "learning_rate": learning_rate,
        "stop_after_epochs": stop_after_epochs,
        "max_num_epochs": max_num_epochs,
        "clip_max_norm": clip_max_norm,
        "validation_fraction": validation_fraction,
        "use_combined_loss": use_combined_loss,
        "show_train_summary": show_train_summary,
        "dataloader_kwargs": {
            "num_workers": num_workers,
            "pin_memory": pin_memory
        }
    }
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check inputs
    if not colours and not luminosity_functions:
        raise ValueError("At least one of colours or luminosity_functions must be True")
    
    # Set up paths
    name = f"{model}_{bands}_{sps}_{spec_type}_{n_bins_lf}_{n_bins_colour}"
    snap_str = str(snap[0])
    cam = camels(model=model, sim_set='LH')
    
    # Define output directories based on what data we're using
    if colours and not luminosity_functions:
        model_out_dir = os.path.join("/disk/xray15/aem2/data/6pams/LH/IllustrisTNG/models/colours_only/", f"{snap_str}")
        plots_out_dir = os.path.join("/disk/xray15/aem2/plots/6pams/LH/IllustrisTNG/test/sbi_plots/colours_only/", f"{snap_str}")
    elif luminosity_functions and not colours:
        model_out_dir = os.path.join("/disk/xray15/aem2/data/6pams/LH/IllustrisTNG/models/lf_only/", f"{snap_str}")
        plots_out_dir = os.path.join("/disk/xray15/aem2/plots/6pams/LH/IllustrisTNG/test/sbi_plots/lf_only/", f"{snap_str}")
    elif colours and luminosity_functions:
        model_out_dir = os.path.join("/disk/xray15/aem2/data/6pams/LH/IllustrisTNG/models/colours_lfs/", f"{snap_str}")
        plots_out_dir = os.path.join("/disk/xray15/aem2/plots/6pams/LH/IllustrisTNG/test/sbi_plots/colours_lfs/", f"{snap_str}")
    
    print("Saving model in", model_out_dir)
    print("Saving plots in", plots_out_dir)
    
    # Check available snapshots
    available_snaps = get_available_snapshots()
    print(f"Available snapshots: {available_snaps}")
    
    # Initialize priors and get data
    prior = initialise_priors(device=device, astro=True, dust=False)
    theta, x = get_theta_x(
        photo_dir=f"/disk/xray15/aem2/data/6pams/LH/IllustrisTNG/photometry",
        spec_type=spec_type,
        model=model,
        snap=snap,
        sps=sps,
        n_bins_lf=n_bins_lf,
        n_bins_colour=n_bins_colour,
        colours=colours,
        luminosity_functions=luminosity_functions,
        device=device,
    )
    
    # Normalize data
    x_all = np.array([np.hstack(_x) for _x in x])
    norm = Normalizer()
    x_all = torch.tensor(
        norm.fit_transform(X=x_all),
        dtype=torch.float32,
        device=device, 
    )
    
    # Import test mask
    test_mask = np.loadtxt("/disk/xray15/aem2/data/6pams/test_mask.txt", dtype=bool)
    
    # Create config string for filenames
    config_str = (f"batch{train_args['training_batch_size']}_"
                 f"lr{train_args['learning_rate']}_"
                 f"epochs{train_args['stop_after_epochs']}_"
                 f"max_num_epochs{train_args['max_num_epochs']}_"
                 f"validation_fraction{train_args['validation_fraction']}_"
                 f"clip_max_norm{train_args['clip_max_norm']}_"
                 f"h{hidden_features}_t{num_transforms}_nn{num_nets}")
    
    # Create directories
    config_plots_dir = os.path.join(plots_out_dir, config_str)
    config_model_dir = os.path.join(model_out_dir, config_str)
    os.makedirs(config_plots_dir, exist_ok=True)
    os.makedirs(config_model_dir, exist_ok=True)
    print("Model will be saved to:", config_model_dir)
    print("Plots will be saved to:", config_plots_dir)
    
    # Save normalizer to config dir
    joblib.dump(norm, os.path.join(config_model_dir, f'data_normaliser_{name}_scaler.save'))
    
    # Create neural networks
    nets = [
        ili.utils.load_nde_sbi(
            engine="NPE",
            model="nsf", 
            hidden_features=hidden_features,
            num_transforms=num_transforms,
        ) for _ in range(num_nets)
    ]
    
    # Set up data loader
    loader = NumpyLoader(
        x=x_all[~test_mask],
        theta=torch.tensor(theta[~test_mask, :], device=device)
    )
    
    # Create inference runner
    runner = InferenceRunner.load(
        backend="sbi",
        engine="NPE",
        prior=prior,
        nets=nets,
        device=device,
        train_args=train_args,
        proposal=None,
        out_dir=config_model_dir,
        name=name,
    )
    
    print('Training with', config_str)
    
    # Train the model
    posterior_ensemble, summaries = runner(loader=loader)
    
    # Create results dictionary
    results = {
        "posterior_ensemble": posterior_ensemble,
        "summaries": summaries,
        "config": {
            "hidden_features": hidden_features,
            "num_transforms": num_transforms,
            "num_nets": num_nets,
            "train_args": train_args,
            "config_str": config_str
        }
    }
    
    # Save results
    with open(os.path.join(config_model_dir, f"{name}_complete.pkl"), "wb") as f:
        pickle.dump(results, f)
    
    # Save posterior ensemble separately
    torch.save(posterior_ensemble, os.path.join(config_model_dir, f"{name}_posterior.pkl"))
    
    # Copy summary file if it exists
    summary_file = os.path.join(model_out_dir, f"{name}summary.json")
    if os.path.exists(summary_file):
        import shutil
        shutil.copy2(summary_file, os.path.join(config_model_dir, f"{name}summary.json"))
    
    # Print paths to saved files
    complete_model_path = os.path.join(config_model_dir, f'{name}_complete.pkl')
    posterior_ensemble_path = os.path.join(config_model_dir, f'{name}_posterior.pkl')
    print(f"Complete model saved to:", complete_model_path)
    print(f"Posterior ensemble saved to:", posterior_ensemble_path)
    
    # Run analysis if requested
    if run_analysis:
        # Plot training diagnostics
        def plot_training_diagnostics(summaries):
            """Plot training diagnostics with loss and overfitting gap"""
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Loss curves
            train_losses = summaries[0]['training_log_probs']
            val_losses = summaries[0]['validation_log_probs']
            epochs = range(len(train_losses))
            
            ax1.plot(epochs, train_losses, '-', label='Training', color='blue')
            ax1.plot(epochs, val_losses, '--', label='Validation', color='red')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Log probability')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Overfitting gap
            gap = np.array(train_losses) - np.array(val_losses)
            ax2.plot(epochs, gap, '-', color='purple')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss difference')
            ax2.set_title('Overfitting Gap')
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
        
        # Plot ensemble training
        def plot_ensemble_training(summaries):
            """Plot training curves for each ensemble member"""
            fig, ax = plt.subplots(1, 1, figsize=(6,4))
            c = list(mcolors.TABLEAU_COLORS)
            for i, m in enumerate(summaries):
                ax.plot(m['training_log_probs'], ls='-', label=f"{i}_train", c=c[i])
                ax.plot(m['validation_log_probs'], ls='--', label=f"{i}_val", c=c[i])
            ax.set_xlim(0)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Log probability')
            ax.legend()
            return fig
        
        # Save training plots
        fig1 = plot_training_diagnostics(summaries)
        plt.savefig(os.path.join(config_plots_dir, f'training_analysis_{name}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        fig2 = plot_ensemble_training(summaries)
        plt.savefig(os.path.join(config_plots_dir, f'ensemble_training_{name}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create metrics and plots
        metric = PosteriorCoverage(
            num_samples=int(1000),
            sample_method='direct',
            labels=cam.labels,
            plot_list=["coverage", "histogram", "predictions", "tarp", "logprob"], 
        )
        
        # Get test data
        x_test = x_all[test_mask]
        theta_test = theta[test_mask].cpu().numpy()
        
        # Get the metric plots
        plot_types = ["coverage", "histogram", "predictions", "tarp", "logprob"]
        figs = metric(
            posterior=posterior_ensemble,
            x=x_test,
            theta=theta_test,
            signature=f"coverage_{name}_"
        )
        
        # Save and display metric plots
        for fig, plot_type in zip(figs, plot_types):
            if fig is not None and hasattr(fig, 'savefig'):
                plot_path = os.path.join(config_plots_dir, f'{plot_type}_{name}.png')
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
        
        print(f"All plots saved in: {config_plots_dir}")
        
        # Create posterior samples metric
        metric = PosteriorSamples(
            num_samples=int(1e4),
            sample_method="direct",
        )
        
        # Use it to get posterior samples
        psamps = metric(
            posterior=posterior_ensemble,
            x=x_all[test_mask],
            theta=theta[test_mask],
        )
        
        # Calculate percentiles and metrics
        perc = np.percentile(psamps, q=[16, 50, 84], axis=0)
        
        # Calculate RMSE
        rmse = np.sqrt(
            np.sum((theta.cpu().numpy()[test_mask, :] - perc[1, :, :])**2, axis=0) / 
            np.sum(test_mask)
        )
        
        # Mean relative error (epsilon)
        mre = np.sum(
            ((perc[2, :, :] - perc[0, :, :]) / 2) / perc[1, :, :], axis=0
        ) / np.sum(test_mask)
        
        # R-squared
        theta_hat = np.sum(theta.cpu().numpy()[test_mask, :], axis=0) / np.sum(test_mask)
        r2 = 1 - np.sum(
            (theta.cpu().numpy()[test_mask, :] - perc[1, :, :])**2, axis=0
        ) / np.sum(
            (theta.cpu().numpy()[test_mask, :] - theta_hat)**2, axis=0
        )
        
        # Chi-squared
        chi2 = np.sum(
            (theta.cpu().numpy()[test_mask, :] - perc[1, :, :])**2 /
            ((perc[2, :, :] - perc[0, :, :]) / 2)**2, axis=0
        ) / np.sum(test_mask)
        
        # Print and save metrics
        metrics_file = os.path.join(config_plots_dir, f'metrics_{name}.txt')
        with open(metrics_file, 'w') as f:
            for i, param in enumerate(cam.labels):
                print(f"\nMetrics for {param}:")
                print(f"RMSE: {rmse[i]:.4f}")
                print(f"Epsilon: {mre[i]:.4f}")
                print(f"R²: {r2[i]:.4f}")
                print(f"χ²: {chi2[i]:.4f}")
                
                f.write(f"\nMetrics for {param}:\n")
                f.write(f"RMSE: {rmse[i]:.4f}\n")
                f.write(f"Epsilon: {mre[i]:.4f}\n")
                f.write(f"R²: {r2[i]:.4f}\n")
                f.write(f"χ²: {chi2[i]:.4f}\n")
        
        print(f"Metrics saved in: {metrics_file}")
    
    return results

if __name__ == "__main__":
    # Default hyperparameters
    train_sbi_model()