import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    # Hack found here: https://github.com/pymc-devs/pymc/issues/7519#issuecomment-2881720629
    import sys

    sys.modules["_multiprocessing"] = object

    import arviz as az
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import warnings
    from scipy import stats
    import pymc as pm

    warnings.filterwarnings("ignore")
    return az, mo, np, plt, pm, stats


@app.cell
def _(mo):
    scenarios = {
        "Underspending": {
            "description": "In this example, the marketing spend is **only** below the saturation point. This makes it difficult to estimate the assymptotic behavior of the saturation curve.",
            "spending range": (0, 0.22),
            "true": {
                "b": 0.82,
                "c": 0.52,
                "sigma": 0.05,
            },
            "priors": {
                "b": {"mu": 0.51, "sigma": 0.05},
                "c": {"mu": 0.5, "sigma": 0.1},
            },
        },
        "Overspending": {
            "description": "In this example, the marketing spend is above the saturation point and notably where the saturation curve is flat.",
            "spending range": (0.67, 1),
            "true": {
                "b": 0.74,
                "c": 0.49,
                "sigma": 0.04,
            },
            "priors": {
                "b": {"mu": 0.7, "sigma": 0.04},
                "c": {"mu": 0.07, "sigma": 0.5},
            },
        },
    }

    default = {
        "spending range": (0, 1),
        "true": {
            "b": 0.45,
            "c": 0.8,
            "sigma": 0.05,
        },
        "priors": {
            "b": {"mu": 0.5, "sigma": 0.5},
            "c": {"mu": 0.5, "sigma": 0.5},
        },
    }

    scenario = mo.ui.dropdown(scenarios, label="Example Scenario:")
    return default, scenario


@app.cell
def _(default, scenario):
    config = scenario.value if scenario.value else default
    return (config,)


@app.cell
def _(config, mo):
    true_b = mo.ui.slider(
        start=0,
        value=config["true"]["b"],
        stop=1,
        step=0.01,
        label="b:",
        show_value=True,
    )
    true_c = mo.ui.slider(
        start=0,
        value=config["true"]["c"],
        stop=1,
        step=0.01,
        label="c:",
        show_value=True,
    )
    scale = mo.ui.slider(
        start=0.01,
        value=config["true"]["sigma"],
        stop=0.15,
        step=0.01,
        label="sigma:",
        show_value=True,
    )

    n_points = mo.ui.slider(
        start=10,
        stop=100,
        step=1,
        value=50,
        label="Number of data points:",
        show_value=True,
    )
    spend_range = mo.ui.range_slider(
        start=0,
        stop=1,
        step=0.01,
        label="Marketing spend range:",
        value=config["spending range"],
        show_value=True,
        debounce=True,
    )
    levels = mo.ui.slider(
        start=10, stop=50, value=40, step=1, label="Contour levels:", show_value=True
    )
    contourf = mo.ui.checkbox(label="Contourf", value=False)

    include_prior = mo.ui.checkbox(value=True, label="Include Prior:")
    b_mu = mo.ui.slider(
        start=0.01,
        value=config["priors"]["b"]["mu"],
        stop=1.5,
        step=0.01,
        label="mu",
        show_value=True,
    )
    b_sigma = mo.ui.slider(
        start=0.01,
        value=config["priors"]["b"]["sigma"],
        stop=1,
        step=0.01,
        label="sigma",
        show_value=True,
    )
    c_mu = mo.ui.slider(
        start=0.01,
        value=config["priors"]["c"]["mu"],
        stop=2,
        step=0.01,
        label="mu",
        show_value=True,
    )
    c_sigma = mo.ui.slider(
        start=0.01,
        value=config["priors"]["c"]["sigma"],
        stop=1,
        step=0.01,
        label="sigma",
        show_value=True,
    )
    return (
        b_mu,
        b_sigma,
        c_mu,
        c_sigma,
        contourf,
        include_prior,
        levels,
        n_points,
        scale,
        spend_range,
        true_b,
        true_c,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    # Geometric intuition for media mix models


    Find the original article [here](https://daniel-saunders-phil.github.io/imagination_machine/posts/geometric-intuition-mmm/) on [Daniel Saunders](
    https://www.linkedin.com/in/dr-daniel-saunders-97239b174/)'s blog.

    In the post, he goes into depth on how different many areas of the likelihood surface look similar and how altering media spends is a route to better explore the surface.
    """
    )
    return


@app.cell
def _(scenario):
    scenario
    return


@app.cell
def _(config, mo):
    description = config.get("description", "")
    mo.md(description)
    return


@app.cell
def _(
    b_mu,
    b_sigma,
    c_mu,
    c_sigma,
    contourf,
    include_prior,
    levels,
    mo,
    n_points,
    scale,
    spend_range,
    true_b,
    true_c,
):
    prior_settings = [
        mo.md("**Priors**"),
        include_prior,
        mo.md("*InverseGamma for b* (x-axis)"),
        b_mu,
        b_sigma,
        mo.md("*InverseGamma for c* (y-axis)"),
        c_mu,
        c_sigma,
    ]

    controls = mo.vstack(
        [
            mo.vstack(
                [
                    mo.md("**True parameters:**"),
                    true_b,
                    true_c,
                    scale,
                ]
            ),
            mo.vstack(prior_settings),
            mo.vstack(
                [
                    mo.md("**Data:**"),
                    n_points,
                    spend_range,
                ]
            ),
            mo.vstack(
                [
                    mo.md("**Surface Settings:**"),
                    levels,
                    contourf,
                ]
            ),
        ],
    )
    return (controls,)


@app.cell
def _(
    controls,
    likelihood_surface,
    mo,
    plot_saturation_curve,
    plot_surface,
    plt,
    posterior_surface,
    prior_surface,
):
    fig, axes = plt.subplots(ncols=3, figsize=(10, 4), sharex=True, sharey=True)

    plot_surface(prior_surface, axes[0]).set(title="prior")
    plot_surface(likelihood_surface, axes[1]).set(title="likelihood", ylabel="")
    plot_surface(posterior_surface, axes[2]).set(title="posterior", ylabel="")
    fig.tight_layout()
    fig.suptitle("")

    curve_fig, ax = plt.subplots(figsize=(8, 4))
    plot_saturation_curve(ax)
    curve_fig.suptitle("")

    mo.hstack(
        [
            controls,
            mo.vstack(
                [
                    mo.as_html(axes[0]),
                    mo.as_html(ax),
                ]
            ),
        ],
        widths=[6, 10],
        gap=0.5,
    )
    return


@app.cell
def _(
    az,
    b_mu,
    b_sigma,
    c_mu,
    c_sigma,
    contourf,
    include_prior,
    levels,
    n_points,
    np,
    pm,
    scale,
    spend_range,
    stats,
    true_b,
    true_c,
):
    def saturation(x, b, c):
        return b * np.tanh(x / (b * c))

    # plot
    np.random.seed(42)

    x_mu = np.linspace(0, 1)
    start, stop = spend_range.value
    x = np.linspace(start, stop, n_points.value)

    # pick some parameters to be the true values

    b = true_b.value
    c = true_c.value

    # push through the saturation function
    mu = saturation(x=x_mu, b=b, c=c)

    y = np.random.normal(loc=saturation(x, b=b, c=c), scale=scale.value)

    with pm.Model() as model:
        _b_prior = pm.InverseGamma("b", mu=b_mu.value, sigma=b_sigma.value)
        _c_prior = pm.InverseGamma("c", mu=c_mu.value, sigma=c_sigma.value)
        pm.Normal(
            "y_obs",
            mu=saturation(x=x_mu, b=_b_prior, c=_c_prior),
            sigma=scale.value,
        )

    curve_samples = pm.sample_prior_predictive(model=model).prior["y_obs"]
    curve_hdi = az.hdi(curve_samples)
    curve_bounds = curve_hdi.y_obs.to_series().unstack()

    def plot_saturation_curve(ax):
        ax.plot(x_mu, mu, color="tab:blue", label="true curve")
        ax.plot(x, y, "o", color="tab:cyan")
        ax.set(xlabel="Marketing spend", ylabel="Sales", title="Simulated data")
        ax.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.25))

        if include_prior.value:
            ax.fill_between(
                x_mu,
                curve_bounds["lower"],
                curve_bounds["higher"],
                color="C1",
                alpha=0.5,
                label="prior predictive interval",
            )
            ax.plot(
                x_mu,
                saturation(x_mu, b_mu.value, c_mu.value),
                color="C1",
                label="prior mean",
            )

        ax.legend()
        return ax

    ### second plot ###
    # build a 2d grid of parameter values

    bs = np.linspace(0, 1.25, 250)
    cs = np.linspace(0, 1.25, 250)

    B, C = np.meshgrid(bs, cs)

    # broadcasting tricks

    xs = np.expand_dims(x, axis=(1, 2))

    mus = saturation(x=xs, b=B, c=C)
    ys = np.expand_dims(y, axis=(1, 2))

    # compute loglikelihood and sum

    likelihood_surface = stats.norm(loc=mus, scale=scale.value).logpdf(ys).sum(axis=0)

    prior_surface = np.zeros_like(likelihood_surface)

    if include_prior.value:
        b_prior = pm.InverseGamma.dist(mu=b_mu.value, sigma=b_sigma.value)
        prior_surface += pm.logp(b_prior, B).eval()

        c_prior = pm.InverseGamma.dist(mu=c_mu.value, sigma=c_sigma.value)
        prior_surface += pm.logp(c_prior, C).eval()

    posterior_surface = likelihood_surface + prior_surface

    if not include_prior.value:
        prior_surface *= np.nan

    def plot_surface(surface, ax):
        contour_func = ax.contour if not contourf.value else ax.contourf

        contour_func(B, C, surface, levels=levels.value)
        ax.plot(b, c, "o")
        ax.annotate(
            "true value",
            (b, c),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=8,
        )
        ax.set(
            xlabel="b (saturation point)",
            ylabel="c (customer acquisition cost)",
        )
        if include_prior.value:
            ax.plot(b_mu.value, c_mu.value, "o")
            ax.annotate(
                "prior mean",
                (b_mu.value, c_mu.value),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=8,
            )
        return ax

    return (
        likelihood_surface,
        plot_saturation_curve,
        plot_surface,
        posterior_surface,
        prior_surface,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
