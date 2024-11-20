from cartopy import crs as ccrs
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely.geometry import Point
import numpy as np
from matplotlib.colors import LogNorm


def fig_setup(grounding_line, study_area, figsize):
    sps = ccrs.SouthPolarStereo()

    plt.figure(figsize=figsize)
    ax = plt.axes(projection=sps)

    if grounding_line is not None:
        ax.add_geometries(
            [grounding_line], crs=sps, facecolor="none", edgecolor="black"
        )
    if study_area is not None:
        ax.add_geometries(
            [study_area], crs=sps, facecolor="none", edgecolor="black", linestyle="--"
        )

    ax.gridlines(
        draw_labels=True,
        x_inline=False,
        y_inline=False,
        dms=False,
        rotate_labels=False,
        zorder=10,
    )

    return ax


def plot_df(
    df,
    val,
    categorical=False,
    title=None,
    cmap="viridis",
    cats=None,
    grounding_line=None,
    study_area=None,
    vmin=None,
    vmax=None,
    figsize=None,
):
    ax = fig_setup(grounding_line, study_area, figsize)

    if not categorical:
        plt.scatter(df["x"], df["y"], s=0.5, c=val, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar()
    else:
        labs = list(cats.keys())
        for i, lab in enumerate(labs):
            idx = df[val] == lab
            plt.scatter(
                df["x"][idx],
                df["y"][idx],
                s=0.5,
                color=cats[lab],
                label=lab,
            )
        plt.legend(frameon=False)

    plt.xlim(min(df["x"]), max(df["x"]))
    plt.ylim(min(df["y"]), max(df["y"]))

    plt.title(title)
    plt.show()


def plot_predictions(df, preds, grounding_line, study_area, figsize):
    plot_df(
        df,
        preds,
        title="Model predictions (1=thawed, 0=frozen)",
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        grounding_line=grounding_line,
        study_area=study_area,
        figsize=figsize,
    )


def plot_img(
    img,
    extent,
    cmap,
    logscale,
    grounding_line,
    study_area,
    vmin,
    title,
    colorlab,
    figsize,
):
    ax = fig_setup(grounding_line, study_area, figsize)
    sps = ccrs.SouthPolarStereo()

    if logscale:
        norm = LogNorm(vmin=vmin)
        vmin = None
    else:
        norm = None
    m = ax.imshow(
        img,
        origin="upper",
        extent=extent,
        transform=sps,
        vmin=vmin,
        cmap=cmap,
        norm=norm,
    )

    plt.title(title)
    plt.colorbar(m, label=colorlab)
    plt.show()


def plot_grid(
    df,
    xy,
    vals,
    intersect,
    cmap,
    grounding_line,
    study_area,
    vmin,
    vmax,
    levels,
    title,
    figsize,
):
    ax = fig_setup(grounding_line, study_area, figsize)

    if intersect is not None:
        assert intersect in ["grounding", "study"]
        intersect_poly = grounding_line if intersect == "grounding" else study_area
        for i in tqdm(range(vals.shape[0])):
            for j in range(vals.shape[1]):
                if not Point(xy[0][i, j], xy[1][i, j]).within(study_area):
                    vals[i, j] = np.nan

    c = plt.contourf(
        xy[0], xy[1], vals, cmap="coolwarm", vmin=vmin, vmax=vmax, levels=20
    )
    plt.colorbar(c)

    plt.xlim(min(df["x"]), max(df["x"]))
    plt.ylim(min(df["y"]), max(df["y"]))

    plt.title(title)
    plt.show()
