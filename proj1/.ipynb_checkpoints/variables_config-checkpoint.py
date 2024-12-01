import os

# In variables_config.py
def get_config(dataset="CV", simulation="IllustrisTNG"):
    """
    Retrieve configuration settings dynamically based on the dataset and simulation.
    """
    # Define the base directory based on dataset and simulation choice
    base_dir = f"/home/jovyan/camels/proj1/{dataset}_set/{dataset}_outputs"
    input_dir = f"/home/jovyan/Data/Photometry/{simulation}/L25n256/{dataset}"
    plots_dir = {
        "UVLFs": {
            "intrinsic": os.path.join(base_dir, "plots", simulation, "UVLFs", "intrinsic"),
            "attenuated": os.path.join(base_dir, "plots", simulation, "UVLFs", "attenuated")
        },
        "colours": {
            "intrinsic": os.path.join(base_dir, "plots", simulation, "colours", "intrinsic"),
            "attenuated": os.path.join(base_dir, "plots", simulation, "colours", "attenuated")
        }
    }    
    # LF data directories with correct structure
    lf_data_dir = {
        "intrinsic": {
            "GALEX": f"{base_dir}/LFs/{simulation}/intrinsic/GALEX",
            "UV1500": f"{base_dir}/LFs/{simulation}/intrinsic/UV1500"
        },
        "attenuated": {
            "GALEX": f"{base_dir}/LFs/{simulation}/attenuated/GALEX",
            "UV1500": f"{base_dir}/LFs/{simulation}/attenuated/UV1500"
        }
    }
    
    # Color data directories with correct structure
    colour_data_dir = {
        "intrinsic": f"{base_dir}/colours/{simulation}/intrinsic",
        "attenuated": f"{base_dir}/colours/{simulation}/attenuated"
    }

    # Define paths and limits for the simulation-specific configurations
    param_info_file = f"Data/Sims/{simulation}/{dataset}/CosmoAstroSeed_{simulation}_L25n256_{dataset}.txt"
    redshift_values = {
        '044': {'redshift': 2.00, 'label': 'z2.0'},
        '052': {'redshift': 1.48, 'label': 'z1.5'},
        '060': {'redshift': 1.05, 'label': 'z1.0'},
        '086': {'redshift': 0.10, 'label': 'z0.1'}
    }

    # GALEX magnitude limits
    mag_limits = {
        '''
        "GALEX_FUV": 24.8,  # From https://iopscience.iop.org/article/10.1086/520512/pdf
        "GALEX_NUV": 24.4
        '''
        "GALEX_FUV": 27,  
        "GALEX_NUV": 27
    }
    
    # Filter definitions
    filters = {
        "intrinsic": ["UV1500", "GALEX FUV", "GALEX NUV"],
        "attenuated": ["GALEX FUV", "GALEX NUV"]
    }

    # Parameters
    #uvlf_limits = (-24, -16)
    uvlf_limits = (-27, -16)
    n_bins_lf = 13# 12 bins!
    colour_limits = (-0.5, 3.5)
    n_bins_colour = 13
    
    colour_pairs = [("GALEX FUV", "GALEX NUV")]

    # Return the complete configuration dictionary
    return {
        "base_dir": base_dir,
        "input_dir": input_dir,
        "plots_dir": plots_dir,
        "param_info_file": param_info_file,
        "redshift_values": redshift_values,
        "lf_data_dir": lf_data_dir,
        "colour_data_dir": colour_data_dir,
        "uvlf_limits": uvlf_limits,
        "n_bins_lf": n_bins_lf,
        "colour_limits": colour_limits,
        "n_bins_colour": n_bins_colour,
        "mag_limits": mag_limits,
        "filters": filters,
        "colour_pairs": colour_pairs
    }
