import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
from patsy import dmatrix
import pandas as pd
import json


PARTIES = [
    "conservative",
    "labour",
    "reform",
    "liberal_democrats",
    "green",
    "plaid_cymru",
    "snp",
    "others",
]
PARTY_COLORS = {
    "conservative": "blue",
    "labour": "red",
    "reform": "lightblue",
    "liberal_democrats": "orange",
    "green": "green",
    "plaid_cymru": "darkgreen",
    "snp": "yellow",
    "others": "gray",
}


def no_pooling_model(df: pd.DataFrame):
    """
    Hierarchical model with pollster bias estimation.
    Each party gets its own independent trend, and each pollster has party-specific biases.
    """
    x = df["date"].values.astype("datetime64[D]").astype(float)  # convert to float days

    num_months = df["date"].dt.to_period("M").nunique()
    print(f"Number of unique months in data: {num_months}")

    # Create B-spline basis matrix using patsy
    num_knots = num_months
    knots = np.linspace(x.min(), x.max(), num_knots + 2)[1:-1]  # interior knots
    B = dmatrix(
        f"bs(x, knots=knots, degree=3, include_intercept=True) - 1",
        {"x": x, "knots": knots},
        return_type="matrix",
    )
    B = np.asarray(B)
    num_basis = B.shape[1]
    num_parties = len(PARTIES)
    n_obs = len(df)

    # Create pollster indices
    pollsters = df["pollster"].unique()
    pollster_map = {p: i for i, p in enumerate(pollsters)}
    pollster_idx = df["pollster"].map(pollster_map).values
    num_pollsters = len(pollsters)
    print(f"Number of pollsters: {num_pollsters}")

    # Stack observations for all parties
    y_data = np.column_stack(
        [df[party].values for party in PARTIES]
    )  # shape: (n_obs, num_parties)

    # Build PyMC model
    with pm.Model() as spline_model:
        # Prior on spline coefficients (random walk for smoothness) - separate for each party
        # Shape: (num_parties, num_basis)
        sigma_coef = pm.HalfNormal("sigma_coef", sigma=0.5, shape=num_parties)
        delta = pm.Normal(
            "delta", mu=0, sigma=sigma_coef[:, None], shape=(num_parties, num_basis)
        )
        coefs = pm.Deterministic("coefs", pm.math.cumsum(delta, axis=1))

        # Mean function for each party (underlying true support)
        # B @ coefs.T gives shape (n_obs, num_parties)
        mu_true = pm.Deterministic("mu_true", pm.math.dot(B, coefs.T))

        # Hierarchical pollster bias: each pollster has a bias for each party
        # Hyperpriors for pollster bias
        sigma_pollster = pm.HalfNormal("sigma_pollster", sigma=2.0, shape=num_parties)

        # Pollster-specific biases: shape (num_pollsters, num_parties)
        pollster_bias = pm.Normal(
            "pollster_bias",
            mu=0,
            sigma=sigma_pollster,
            shape=(num_pollsters, num_parties),
        )

        # Add pollster bias to the mean function
        # pollster_bias[pollster_idx] has shape (n_obs, num_parties)
        mu = pm.Deterministic("mu", mu_true + pollster_bias[pollster_idx, :])

        # Likelihood - separate sigma for each party
        sigma = pm.HalfNormal("sigma", sigma=1.0, shape=num_parties)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_data)

        # Sample
        trace = pm.sample(
            1000, tune=1000, target_accept=0.9, random_seed=42, nuts_sampler="nutpie"
        )

    # Extract results for each party
    party_results = {}
    mu_samples = trace.posterior[
        "mu_true"
    ].values  # shape: (chains, draws, n_obs, num_parties)
    mu_samples = mu_samples.reshape(
        -1, n_obs, num_parties
    )  # shape: (total_samples, n_obs, num_parties)

    for i, party in enumerate(PARTIES):
        mu_party = mu_samples[:, :, i]  # shape: (total_samples, n_obs)
        mu_mean = mu_party.mean(axis=0)
        mu_hdi = np.percentile(mu_party, [2.5, 97.5], axis=0)
        party_results[party] = {"mean": mu_mean, "hdi": mu_hdi}

    # Extract pollster biases
    pollster_bias_samples = trace.posterior["pollster_bias"].values
    pollster_bias_samples = pollster_bias_samples.reshape(
        -1, num_pollsters, num_parties
    )

    pollster_biases = {}
    for i, pollster in enumerate(pollsters):
        pollster_biases[pollster] = {}
        for j, party in enumerate(PARTIES):
            bias_samples = pollster_bias_samples[:, i, j]
            pollster_biases[pollster][party] = {
                "mean": float(bias_samples.mean()),
                "std": float(bias_samples.std()),
                "hdi": [float(x) for x in np.percentile(bias_samples, [2.5, 97.5])],
            }

        # Count number of polls for this pollster
        num_polls = int((df["pollster"] == pollster).sum())
        pollster_biases[pollster]["num_polls"] = num_polls

        # Compute overall pollster bias (RMS across parties)
        party_biases = [pollster_biases[pollster][party]["mean"] for party in PARTIES]
        overall_bias = float(np.sqrt(np.mean(np.array(party_biases) ** 2)))

        # Compute pollster rating based on overall bias
        if overall_bias < 0.5:
            rating = "A+"
        elif overall_bias < 1.0:
            rating = "A"
        elif overall_bias < 1.5:
            rating = "B+"
        elif overall_bias < 2.0:
            rating = "B"
        elif overall_bias < 2.5:
            rating = "C+"
        elif overall_bias < 3.0:
            rating = "C"
        elif overall_bias < 4.0:
            rating = "D"
        else:
            rating = "F"

        pollster_biases[pollster]["overall_bias"] = overall_bias
        pollster_biases[pollster]["rating"] = rating

    return party_results, trace, pollster_biases


def main():
    df = pd.read_parquet("data/processed/uk_polling_data.parquet")
    df = df.dropna(subset=["date"])
    df = df.loc[~df.sample_size.isna()]

    results, trace, pollster_biases = no_pooling_model(df)

    # Save results as JSON (convert numpy arrays to lists)
    results_json = {}
    for party, res in results.items():
        results_json[party] = {
            "mean": res["mean"].tolist(),
            "hdi_lower": res["hdi"][0].tolist(),
            "hdi_upper": res["hdi"][1].tolist(),
        }

        plt.plot(df["date"], res["mean"], label=f"{party}", color=PARTY_COLORS[party])
        plt.fill_between(
            df["date"],
            res["hdi"][0],
            res["hdi"][1],
            color=PARTY_COLORS[party],
            alpha=0.3,
        )
        plt.scatter(df["date"], df[party], color=PARTY_COLORS[party], alpha=0.5)

    # Write dataframe to JSON file
    df_json = df.copy()
    df_json["date"] = df_json["date"].astype(str)  # Convert dates to strings for JSON
    df_json.to_json("data/processed/polling_data.json", orient="records", indent=2)

    # Write results to JSON file
    with open("data/processed/polling_results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    # Write pollster biases to JSON file
    with open("data/processed/pollster_biases.json", "w") as f:
        json.dump(pollster_biases, f, indent=2)

    plt.xlabel("Date")
    plt.ylabel("Support (%)")
    plt.legend()
    plt.savefig("data/processed/polling_trends.png")


if __name__ == "__main__":
    main()
