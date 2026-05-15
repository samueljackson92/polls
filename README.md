# UK Polling Aggregator 🗳️

A Bayesian statistical model for aggregating and visualizing UK opinion polling data. This project scrapes polling data from Wikipedia, fits a Bayesian B-spline regression model to estimate smooth trends for each political party, and presents the results in an interactive web dashboard.

## Overview

Rather than simply averaging polls or using traditional smoothing methods (like LOESS), this project uses a Bayesian approach to estimate polling trends while accounting for uncertainty. Each party's support is modeled independently using B-spline basis functions, with posterior distributions estimated via Markov Chain Monte Carlo (MCMC) sampling.

## Methodology

### Statistical Model

- **Model type:** Bayesian B-spline regression with no pooling across parties
- **Spline knots:** Approximately one knot per month (adaptive to data)
- **Likelihood:** Normal distribution with party-specific variance
- **Inference:** MCMC sampling using PyMC with the NUTS sampler (via nutpie)

### Key Advantages

- **Proper uncertainty quantification:** Unlike simple averages, the model provides credible intervals that reflect both sampling uncertainty and model uncertainty
- **Smooth trends:** B-splines provide flexible, smooth curves that adapt to the data
- **Separate party modeling:** Each party gets its own independent trend

### Known Limitations

- All polls are treated as equally reliable (no weighting for pollster house effects or quality)
- Parties are modeled independently without accounting for correlations
- Vote shares may not sum to 100% across parties
- This is a descriptive model, not a predictive/forecasting model

## Project Structure

```
polls/
├── extract_polls.py        # Web scraper for Wikipedia polling data
├── main.py                 # Bayesian model fitting and result generation
├── index.html              # Interactive visualization dashboard
├── data/
│   ├── raw/               # Raw scraped HTML
│   └── processed/         # Processed parquet/JSON files
├── pyproject.toml         # Project dependencies
└── README.md              # This file
```

## Installation

This project uses Python 3.12+ and uv for dependency management.

```bash
# Clone the repository
git clone <repository-url>
cd polls

# Install dependencies with uv
uv sync
```

## Usage

### 1. Scrape polling data

```bash
python extract_polls.py
```

This scrapes the latest polling data from Wikipedia and saves it as `data/processed/uk_polling_data.parquet`.

### 2. Fit the Bayesian model

```bash
python main.py
```

This:
- Loads the polling data
- Fits the Bayesian B-spline model
- Exports results to `data/processed/polling_results.json` and `data/processed/polling_data.json`

### 3. View the dashboard

Open `index.html` in a web browser. The dashboard reads the JSON files and renders an interactive visualization.

## Data Source

All polling data is sourced from: https://en.wikipedia.org/wiki/Opinion_polling_for_the_next_United_Kingdom_general_election

Wikipedia aggregates polls from various organizations including YouGov, Ipsos MORI, Savanta, Redfield & Wilton, and others.

## Future Improvements

Potential enhancements for future iterations:

- **Multinomial regression** to ensure vote shares sum to 100%
- **Predictive modeling** with proper forecasting methodology
- **Temporal correlation modeling** between parties

## License

This project is for educational and research purposes. Polling data is sourced from Wikipedia and belongs to the respective polling organizations.

## Disclaimer

This is a statistical model for analyzing polling trends and should **not** be interpreted as a prediction or forecast of election results. Polls can be volatile and may not accurately reflect final election outcomes.