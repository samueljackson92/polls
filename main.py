import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
from patsy import dmatrix
import pandas as pd


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
    "reform": "cyan",
    "liberal_democrats": "orange",
    "green": "green",
    "plaid_cymru": "darkgreen",
    "snp": "yellow",
    "others": "gray",
}


def single_model(df: pd.DataFrame, party: str = "labour"):
    x = df["date"].values.astype("datetime64[D]").astype(float)  # convert to float days
    y = df[party].values  # target variable (e.g., Labour support)

    num_months = df["date"].dt.to_period("M").nunique()
    print(f"Number of unique months in data: {num_months}")

    # Create B-spline basis matrix using patsy
    num_knots = num_months // 3  # heuristic: one knot every 3 months
    knots = np.linspace(x.min(), x.max(), num_knots + 2)[1:-1]  # interior knots
    B = dmatrix(
        f"bs(x, knots=knots, degree=3, include_intercept=True) - 1",
        {"x": x, "knots": knots},
        return_type="matrix",
    )
    B = np.asarray(B)
    num_basis = B.shape[1]

    # Build PyMC model
    with pm.Model() as spline_model:
        # Prior on spline coefficients (random walk for smoothness)
        sigma_coef = pm.HalfNormal("sigma_coef", sigma=1.0)
        delta = pm.Normal("delta", mu=0, sigma=sigma_coef, shape=num_basis)
        coefs = pm.Deterministic("coefs", pm.math.cumsum(delta))

        # Mean function
        mu = pm.Deterministic("mu", pm.math.dot(B, coefs))

        # Likelihood
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        # Sample
        trace = pm.sample(
            1000, tune=1000, target_accept=0.9, random_seed=42, nuts_sampler="nutpie"
        )
    return trace


def independent_model(df: pd.DataFrame):
    party_results = {}
    for party in PARTIES:
        x = (
            df["date"].values.astype("datetime64[D]").astype(float)
        )  # convert to float days
        y = df[party].values  # target variable for the party

        trace = single_model(df, party)
        mu_samples = trace.posterior["mu"].values.reshape(-1, len(df))
        mu_mean = mu_samples.mean(axis=0)
        mu_hdi = np.percentile(mu_samples, [2.5, 97.5], axis=0)
        party_results[party] = {"mean": mu_mean, "hdi": mu_hdi}
    return party_results


def main():
    df = pd.read_parquet("uk_polling_data.parquet")
    df = df.dropna(subset=["date"])

    results = independent_model(df)
    plt.figure(figsize=(10, 5))
    for party, res in results.items():
        plt.plot(
            df["date"], res["mean"], label=f"{party} mean", color=PARTY_COLORS[party]
        )
        plt.fill_between(
            df["date"],
            res["hdi"][0],
            res["hdi"][1],
            color=PARTY_COLORS[party],
            alpha=0.3,
            label=f"{party} 95% HDI",
        )
        plt.scatter(
            df["date"], df[party], color=PARTY_COLORS[party], alpha=0.5, label="Data"
        )

    plt.xlabel("Date")
    plt.ylabel("Support (%)")
    plt.legend()
    plt.savefig("polling_trends.png")
    plt.show()


if __name__ == "__main__":
    main()
