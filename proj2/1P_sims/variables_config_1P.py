# config.py

import os

# Base directories
base_dir = "/disk/xray15/aem2/data/28pams/IllustrisTNG/1P"
input_dir = os.path.join(base_dir, "photometry")
plots_dir = "/disk/xray15/aem2/plots/28pams/IllustrisTNG/1P"
param_info_file = "/disk/xray15/aem2/data/28pams/Info_IllustrisTNG_L25n256_28params.txt"

# Redshift mappings
redshift_values = {
    '044': {'redshift': 2.00, 'label': 'z2.0'},
    '052': {'redshift': 1.48, 'label': 'z1.5'},
    '060': {'redshift': 1.05, 'label': 'z1.0'},
    '086': {'redshift': 0.10, 'label': 'z0.1'}
}

'''
# Output directories - can use for base directory when generating UVLFs and Colour txt as well as plots
def lf_output_dir(base_dir):
    lf_out_dir = {
        "attenuated": {
            "GALEX": f"{base_dir}/LFs/attenuated/GALEX"  # Attenuated GALEX
            },
        "intrinsic": {
            "UV1500": f"{base_dir}/LFs/intrinsic/UV1500",  # Pure rest-frame
            "GALEX": f"{base_dir}/LFs/intrinsic/GALEX"     # Rest-frame through GALEX filters
            }
        }
    return lf_out_dir

def colour_output_dir(base_dir):

    colour_out_dir= {
        "attenuated": {
            "GALEX": f"{base_dir}/colours/attenuated/GALEX"
            },
        "intrinsic": {
            "GALEX": f"{base_dir}/colours/intrinsic/GALEX"
            }
        }
    return colour_out_dir
'''

# Output directories 
lf_data_dir = {
    "attenuated": {
        "GALEX": f"{base_dir}/LFs/attenuated/GALEX"
    },
    "intrinsic": {
        "UV1500": f"{base_dir}/LFs/intrinsic/UV1500",
        "GALEX": f"{base_dir}/LFs/intrinsic/GALEX"
    }
}

colour_data_dir = {
    "attenuated": {
        "GALEX": f"{base_dir}/colours/attenuated/GALEX",
        "GALEX_FUV-NUV": f"{base_dir}/colours/attenuated/GALEX_FUV-NUV"
    },
    "intrinsic": {
        "GALEX": f"{base_dir}/colours/intrinsic/GALEX",
        "GALEX_FUV-NUV": f"{base_dir}/colours/intrinsic/GALEX_FUV-NUV"
    }
}

# Parameters
uvlf_limits = (-24, -16)
uvlf_nbins = 11
colour_limits = (-0.5, 3.5)
colour_nbins = 20

# Magnitude limits (DIS) from https://iopscience.iop.org/article/10.1086/520512/pdf
NUV_mag_lim = 24.4
FUV_mag_lim = 24.8

# GALEX magnitude limits
mag_limits = {
    "GALEX_FUV": 24.8,  # From https://iopscience.iop.org/article/10.1086/520512/pdf
    "GALEX_NUV": 24.4
}

# Filter definitions
filters = {
    "intrinsic": ["UV1500", "GALEX FUV", "GALEX NUV"],
    "attenuated": ["GALEX FUV", "GALEX NUV"]
}

# colour pairs we want to handle.
colour_pairs = [("GALEX FUV", "GALEX NUV")]
