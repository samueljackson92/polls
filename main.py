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
    No-pooling model: each party gets its own independent trend.
    All parties are fit simultaneously in one model.
    """
    x = df["date"].values.astype("datetime64[D]").astype(float)  # convert to float days

    num_months = df["date"].dt.to_period("M").nunique()
    print(f"Number of unique months in data: {num_months}")

    # Create B-spline basis matrix using patsy
    num_knots = num_months  # heuristic: one knot every 3 months
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

    # Stack observations for all parties
    y_data = np.column_stack(
        [df[party].values for party in PARTIES]
    )  # shape: (n_obs, num_parties)

    # Build PyMC model
    with pm.Model() as spline_model:
        # Prior on spline coefficients (random walk for smoothness) - separate for each party
        # Shape: (num_parties, num_basis)
        sigma_coef = pm.HalfNormal("sigma_coef", sigma=1.0, shape=num_parties)
        delta = pm.Normal(
            "delta", mu=0, sigma=sigma_coef[:, None], shape=(num_parties, num_basis)
        )
        coefs = pm.Deterministic("coefs", pm.math.cumsum(delta, axis=1))

        # Mean function for each party
        # B @ coefs.T gives shape (n_obs, num_parties)
        mu = pm.Deterministic("mu", pm.math.dot(B, coefs.T))

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
        "mu"
    ].values  # shape: (chains, draws, n_obs, num_parties)
    mu_samples = mu_samples.reshape(
        -1, n_obs, num_parties
    )  # shape: (total_samples, n_obs, num_parties)

    for i, party in enumerate(PARTIES):
        mu_party = mu_samples[:, :, i]  # shape: (total_samples, n_obs)
        mu_mean = mu_party.mean(axis=0)
        mu_hdi = np.percentile(mu_party, [2.5, 97.5], axis=0)
        party_results[party] = {"mean": mu_mean, "hdi": mu_hdi}

    return party_results, trace


def main():
    df = pd.read_parquet("data/processed/uk_polling_data.parquet")
    df = df.dropna(subset=["date"])
    df = df.loc[~df.sample_size.isna()]

    results, trace = no_pooling_model(df)

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

    plt.xlabel("Date")
    plt.ylabel("Support (%)")
    plt.legend()
    plt.savefig("data/processed/polling_trends.png")


if __name__ == "__main__":
    main()
