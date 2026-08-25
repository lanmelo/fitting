"""Example script showing how to fit dissociation curves."""

# pylint: disable=wrong-import-position,invalid-name,line-too-long

import os
import sys

# Force JAX to use CPU to avoid CUDA initialization errors on nodes without GPUs
# unless explicitly overridden by user environment variables
if "JAX_PLATFORM_NAME" not in os.environ:
    os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax.numpy as jnp
import pandas as pd

# Add the src folder to path if running this script directly without installing the package
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
)
from fitting import SingleExponentialIntervalModel, fit_curves

if __name__ == "__main__":
    # Define file paths
    # Note: Update these paths to match your directory structure if running elsewhere
    data_path = "/Users/lucas/Library/CloudStorage/GoogleDrive-lanmelo@stanford.edu/Shared drives/FordyceLab/Lab_Member_Files/Lucas/Experiments/01_MAX_STAMMPseq/Renee & Lucas: STAMMP_Seq/Analysis/sherlock_koff_fits6/final_kinetic_counts.csv.gz"
    outfile = "koffs.csv.gz"

    if not os.path.exists(data_path):
        print(f"Could not find data file at {data_path}")
        print(
            "Please run this script from the workspace directory or check the path."
        )
        sys.exit(1)

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(
        data_path,
        index_col=["replicate", "variant", "barcode", "background", "motif"],
    )

    # Process normalization spike-in vectors
    spikein = df.xs("spike", level="variant").droplevel(
        ["barcode", "background", "motif"], axis=0
    )
    assert isinstance(spikein, pd.DataFrame)
    norm_spikein = spikein.div(spikein.iloc[:, -1], axis=0)
    norm_dict = dict(zip(norm_spikein.index, norm_spikein.to_numpy()))

    # Filter out spike-ins and empty rows
    filt_df = df.drop("spike", level="variant")
    filt_df = filt_df[filt_df.sum(axis=1) > 0]

    # If --test is provided, only fit a subset for fast validation
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Test mode: Slicing dataframe to first 500 rows.")
        filt_df = filt_df.iloc[:500]

    # Independent variable (timepoints in hours)
    x_data = jnp.pad(jnp.arange(2.0, 24.1, 2.0), (0, 1), constant_values=24.0)

    # Initialize the Single Exponential Interval Model (with bounds concatenated)
    model = SingleExponentialIntervalModel(concat_bound=True)

    # Fit all sequences concurrently using BFGS and JAX acceleration
    print("Fitting dissociation curves...")
    out_df = fit_curves(
        df=filt_df,
        model=model,
        x_data=x_data,
        norm=norm_dict,
        key_level="replicate",
        chunk_size=50_000,
    )

    print("Finished fitting.", flush=True)

    # Save output
    print(f"Saving results to {outfile}...")
    out_df.to_csv(outfile)
    print("Done!")
