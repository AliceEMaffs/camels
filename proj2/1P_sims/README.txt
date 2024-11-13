Alice Eleanor Matthews
Read me for using the code to get UVLF and colour plots and txts from jupyter notebooks

These plots are generated using this code:
https://github.com/AliceEMaffs/camels/blob/main/proj2/1P_sims/plots_all_uvlf_colours.ipynb
which uses txt files from here:
https://github.com/AliceEMaffs/camels/blob/main/proj2/1P_sims/get_all_uvlfs_colours.ipynb
which gets the photometry from the hdf5 files outside of camels dir.

To use the above scripts, you should only need to change the path directories inside
https://github.com/AliceEMaffs/camels/blob/main/proj2/1P_sims/variables_config.py
and you can add additional filters and bands to the dictionaries if required.

The scripts should generate the same file directories as structured below,
 for you to store your LFs and colour txts and plots like this:

'''
plots_dir/ and input_dir structure generated from the plots_all_uvlf_colours.ipynb and get_all_uvlf_colours_ipynb
├── LFs/
│   ├── intrinsic/
│   │   ├── UV1500/
│   │   │   ├── /z0.1
│   │   │   ├── /z1.0
│   │   │   ├── /z1.5
│   │   │   ├── /z2.0
│   │   │   └── parameter_variations/
│   │   │   └── redshift_variations/
│   │   ├── GALEX_FUV/
│   │   │   ├── /z0.1
│   │   │   ├── /z1.0
│   │   │   ├── /z1.5
│   │   │   ├── /z2.0
│   │   │   ├── parameter_variations/
│   │   │   └── redshift_variations/
│   │   └── GALEX_NUV/
│   │       ├── parameter_variations/
│   │       └── redshift_variations/
│   └── attenuated/
│       ├── GALEX_FUV/
│       │   ├── /z0.1
│       │   ├── /z1.0
│       │   ├── /z1.5
│       │   ├── /z2.0
│       │   ├── parameter_variations/
│       │   └── redshift_variations/
│       └── GALEX_NUV/
│           ├── parameter_variations/
│           └── redshift_variations/
└── colours/
    ├── intrinsic/
    │   └── GALEX/
    │       ├── /z0.1
    │       ├── /z1.0
    │       ├── /z1.5
    │       ├── /z2.0
    │       ├── parameter_variations/
    │       └── redshift_variations/
    └── attenuated/
        └── GALEX/
            ├── /z0.1
            ├── /z1.0
            ├── /z1.5
            ├── /z2.0
            ├── parameter_variations/
            └── redshift_variations/
'''