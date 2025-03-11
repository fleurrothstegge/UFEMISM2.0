import os
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from python_src.plotting.colormaps import *


def extract_voronoi_cells(ds):
    """
    # Function that extracts Voronoi cell data from particular mesh file
    """
    # Extract Voronoi cells
    patches = []
    for vi in range(0, len(ds.vi)):
        nVVor = ds.nVVor[vi].values
        VVor = ds.VVor[:nVVor, vi].values
        Vor = ds.Vor[:, VVor - 1].values
        patches.append(Polygon(Vor.T))

    return patches


def plot_UFE_mesh_single(
    patches,
    ds,
    var,
    t,
    mask=[],
    set_limits=None,
    save_to_file=None,
    speedup=None,
    figax=None,
):
    """
    Plots a single variable (var) from the dataset (ds) on the UFE mesh at a specified time (t).
    """

    # Get colormap data
    cmap, norm = get_colormaps()

    # Select time slice
    ds_t = ds.sel(time=t)

    if speedup == True:
        ds_t = ds_t.astype(np.float32)  # Convert everything in ds to float32 to speed things up

    if figax == None:
        # Prepare figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    else:
        fig, ax = figax[0], figax[1]

    # Create patch with variable-dependent colormap data
    p = PatchCollection(patches, cmap=cmap[var], norm=norm[var])

    # If var = BMB or similar, multiply by -1 to get melt rates to show as positive, refreezing negative
    if var in ["BMB", "BMB_inv", "BMB_transition_phase"]:
        cols = [-ds_t[var][vi].values for vi in range(0, len(ds_t.vi))]
    else:
        cols = [ds_t[var][vi].values for vi in range(0, len(ds_t.vi))]

    # Attach data to patch
    p.set_array(cols)

    # Plot cells
    im = ax.add_collection(p)

    try:
        fig.colorbar(
            im,
            ax=ax,
            label=f"{ds_t[var].attrs['long_name']} [{ds_t[var].attrs['units']}]",
            orientation="vertical",
            shrink=0.7,
            pad=0.05,
            extend="both",
        )
    except:
        fig.colorbar(
            im,
            ax=ax,
            label=f"{ds_t[var].attrs['long_name']}",
            orientation="vertical",
            shrink=0.7,
            pad=0.05,
            extend="both",
        )

    # Make up grid of subplot
    ax.set_xlim([ds_t.xmin, ds_t.xmax])
    ax.set_ylim([ds_t.ymin, ds_t.ymax])

    # Mask out specified areas
    if "ocean" in mask:
        ds_mask_oc_flat = xr.where(ds_t["mask"] == 2, 2, np.nan)
        p_oc = PatchCollection(patches, cmap=cmap["mask"], norm=norm["mask"])
        cols_mask = [ds_mask_oc_flat[vi].values for vi in range(0, len(ds_mask_oc_flat.vi))]
        p_oc.set_array(cols_mask)
        ax.add_collection(p_oc)

    if "grounded" in mask:
        ds_mask_gr_flat = xr.where(ds_t["mask"] == 3, 3, np.nan)
        p_gr = PatchCollection(patches, cmap=cmap["mask"], norm=norm["mask"])
        cols_mask = [ds_mask_gr_flat[vi].values for vi in range(0, len(ds_mask_gr_flat.vi))]
        p_gr.set_array(cols_mask)
        ax.add_collection(p_gr)

    if "GL_gr" in mask:
        ds_mask_gl_flat = xr.where(ds_t["mask"] == 5, 5, np.nan)
        p_gl = PatchCollection(patches, cmap=cmap["mask"], norm=norm["mask"])
        cols_mask = [ds_mask_gl_flat[vi].values for vi in range(0, len(ds_mask_gl_flat.vi))]
        p_gl.set_array(cols_mask)
        ax.add_collection(p_gl)

    if "GL_fl" in mask:
        ds_mask_gl_flat = xr.where(ds_t["mask"] == 6, 6, np.nan)
        p_gl = PatchCollection(patches, cmap=cmap["mask"], norm=norm["mask"])
        cols_mask = [ds_mask_gl_flat[vi].values for vi in range(0, len(ds_mask_gl_flat.vi))]
        p_gl.set_array(cols_mask)
        ax.add_collection(p_gl)

    # Zoom in on WAIS
    if set_limits == "WAIS":
        ax.set_xlim([-2000000, 0])
        ax.set_ylim([-1500000, 500000])

    elif set_limits == "WAIS_zoom":
        ax.set_xlim([-1750000, -1250000])
        ax.set_ylim([-750000, -200000])

    ax.set_title(f'Variable "{var}" \n time = {t}')
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect(1)
    ax.grid(True, ls="dashed", alpha=0.3)

    fig.tight_layout()

    if save_to_file:
        # Save figure
        plt.savefig(f"{save_to_file}")

    return ax


# Function that plots difference between two different time slices for one variable for one dataset
def plot_UFE_mesh_single_difference_time(
    patches,
    ds,
    var,
    time_A,
    time_B,
    mask=[],  # Note: if you plot mask here, it plots the mask for at time_A
    vmin=-10,
    vmax=10,
    set_limits=None,
    save_to_file=None,
    figax=None,
):
    """
    This function plots the difference between a variable (var) from the dataset (ds) at
    two different times (time_A and time_B) on the UFE mesh.
    """

    # Get colormap data
    cmap, norm = get_colormaps()

    # Compute difference
    ds_A = ds.sel(time=time_A)
    ds_B = ds.sel(time=time_B)
    ds_diff = ds_A - ds_B

    # Prepare figure
    if figax == None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    else:
        fig, ax = figax[0], figax[1]

    # Create patch with variable-dependent colormap data
    p = PatchCollection(
        patches, cmap="cmo.balance", norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    )

    # If var = BMB or similar, multiply by -1 to ensure: RED means var(time_A) increased wrsp var(time_B)
    if var in ["BMB", "BMB_inv", "BMB_transition_phase"]:
        cols = [-ds_diff[var][vi].values for vi in range(0, len(ds.vi))]
    else:
        cols = [ds_diff[var][vi].values for vi in range(0, len(ds.vi))]

    # Attach data to patch
    p.set_array(cols)

    # Plot cells
    im = ax.add_collection(p)

    # Show colorbar
    fig.colorbar(
        im,
        ax=ax,
        label=f"Difference {var}",
        orientation="vertical",
        shrink=0.7,
        pad=0.05,
        extend="both",
    )

    # Make up grid of subplot
    ax.set_xlim([ds.xmin, ds.xmax])
    ax.set_ylim([ds.ymin, ds.ymax])
    ax.set_aspect(1)
    ax.grid(True)

    # Mask out specified areas
    if "ocean" in mask:
        ds_mask_oc_flat = xr.where(ds_A["mask"] == 2, 2, np.nan)
        p_oc = PatchCollection(patches, cmap=cmap["mask"], norm=norm["mask"])
        cols_mask = [ds_mask_oc_flat[vi].values for vi in range(0, len(ds_mask_oc_flat.vi))]
        p_oc.set_array(cols_mask)
        ax.add_collection(p_oc)

    if "grounded" in mask:
        ds_mask_gr_flat = xr.where(ds_A["mask"] == 3, 3, np.nan)
        p_gr = PatchCollection(patches, cmap=cmap["mask"], norm=norm["mask"])
        cols_mask = [ds_mask_gr_flat[vi].values for vi in range(0, len(ds_mask_gr_flat.vi))]
        p_gr.set_array(cols_mask)
        ax.add_collection(p_gr)

    if "GL_gr" in mask:
        ds_mask_gl_flat = xr.where(ds_A["mask"] == 5, 5, np.nan)
        p_gl = PatchCollection(patches, cmap=cmap["mask"], norm=norm["mask"])
        cols_mask = [ds_mask_gl_flat[vi].values for vi in range(0, len(ds_mask_gl_flat.vi))]
        p_gl.set_array(cols_mask)
        ax.add_collection(p_gl)

    if "GL_fl" in mask:
        ds_mask_gl_flat = xr.where(ds_A["mask"] == 6, 6, np.nan)
        p_gl = PatchCollection(patches, cmap=cmap["mask"], norm=norm["mask"])
        cols_mask = [ds_mask_gl_flat[vi].values for vi in range(0, len(ds_mask_gl_flat.vi))]
        p_gl.set_array(cols_mask)
        ax.add_collection(p_gl)

    # Zoom in on WAIS
    if set_limits == "WAIS":
        ax.set_xlim([-2000000, 0])
        ax.set_ylim([-1500000, 500000])

    elif set_limits == "WAIS_zoom":
        ax.set_xlim([-1750000, -1250000])
        ax.set_ylim([-750000, -200000])

    ax.set_title(f'Difference variable "{var}" \n t = {time_A} minus t = {time_B}')
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect(1)
    ax.grid(True, ls="dashed", alpha=0.3)

    fig.tight_layout()

    if save_to_file:
        # Save figure
        plt.savefig(f"{save_to_file}")

    return


# Function that plots difference between two different variables for one dataset at one time slice
def plot_UFE_mesh_single_difference_var(
    patches,
    ds,
    var_A,
    var_B,
    t,
    mask=[],
    vmin=-10,
    vmax=10,
    set_limits=None,
    save_to_file=None,
    figax=None,
):
    """
    This function plots the difference between two variables (var_A, var_B) from the dataset (ds) at
    time (t) on the UFE mesh.
    """

    # Get colormap data
    cmap, norm = get_colormaps()

    # Compute difference
    if var_A in ["BMB", "BMB_inv", "BMB_transition_phase"]:
        ds_A = -ds[var_A].sel(time=t)
    else:
        ds_A = ds[var_A].sel(time=t)

    if var_B in ["BMB", "BMB_inv", "BMB_transition_phase"]:
        ds_B = -ds[var_B].sel(time=t)
    else:
        ds_B = ds[var_B].sel(time=t)

    ds_diff = ds_A - ds_B

    # Prepare figure
    if figax == None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    else:
        fig, ax = figax[0], figax[1]

    # Create patch with variable-dependent colormap data
    p = PatchCollection(
        patches, cmap="cmo.balance", norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    )

    cols = [ds_diff[vi].values for vi in range(0, len(ds.vi))]

    # Attach data to patch
    p.set_array(cols)

    # Plot cells
    im = ax.add_collection(p)

    # Show colorbar
    fig.colorbar(
        im,
        ax=ax,
        label=f"Difference {var_A} minus {var_B}",
        orientation="vertical",
        shrink=0.7,
        pad=0.05,
        extend="both",
    )

    # Make up grid of subplot
    ax.set_xlim([ds.xmin, ds.xmax])
    ax.set_ylim([ds.ymin, ds.ymax])
    ax.set_aspect(1)
    ax.grid(True)

    # Mask out specified areas
    if "ocean" in mask:
        ds_mask_oc_flat = xr.where(ds.sel(time=t)["mask"] == 2, 2, np.nan)
        p_oc = PatchCollection(patches, cmap=cmap["mask"], norm=norm["mask"])
        cols_mask = [ds_mask_oc_flat[vi].values for vi in range(0, len(ds_mask_oc_flat.vi))]
        p_oc.set_array(cols_mask)
        ax.add_collection(p_oc)

    if "grounded" in mask:
        ds_mask_gr_flat = xr.where(ds.sel(time=t)["mask"] == 3, 3, np.nan)
        p_gr = PatchCollection(patches, cmap=cmap["mask"], norm=norm["mask"])
        cols_mask = [ds_mask_gr_flat[vi].values for vi in range(0, len(ds_mask_gr_flat.vi))]
        p_gr.set_array(cols_mask)
        ax.add_collection(p_gr)

    if "GL_gr" in mask:
        ds_mask_gl_flat = xr.where(ds.sel(time=t)["mask"] == 5, 5, np.nan)
        p_gl = PatchCollection(patches, cmap=cmap["mask"], norm=norm["mask"])
        cols_mask = [ds_mask_gl_flat[vi].values for vi in range(0, len(ds_mask_gl_flat.vi))]
        p_gl.set_array(cols_mask)
        ax.add_collection(p_gl)

    if "GL_fl" in mask:
        ds_mask_gl_flat = xr.where(ds.sel(time=t)["mask"] == 6, 6, np.nan)
        p_gl = PatchCollection(patches, cmap=cmap["mask"], norm=norm["mask"])
        cols_mask = [ds_mask_gl_flat[vi].values for vi in range(0, len(ds_mask_gl_flat.vi))]
        p_gl.set_array(cols_mask)
        ax.add_collection(p_gl)

    # Zoom in on WAIS
    if set_limits == "WAIS":
        ax.set_xlim([-2000000, 0])
        ax.set_ylim([-1500000, 500000])

    elif set_limits == "WAIS_zoom":
        ax.set_xlim([-1750000, -1250000])
        ax.set_ylim([-750000, -200000])

    ax.set_title(f'Difference at time {t} \n variable "{var_A}" minus "{var_B}"')
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect(1)
    ax.grid(True, ls="dashed", alpha=0.3)

    fig.tight_layout()

    if save_to_file:
        # Save figure
        plt.savefig(f"{save_to_file}")

    return


# Function that plots difference between two different variables for one dataset at one time slice
def plot_UFE_mesh_single_difference_ds(
    patches,
    ds_AA,
    ds_BB,
    var,
    t,
    mask=[],  # Note: if you plot mask here, it plots the mask for ds_AA at time t
    vmin=-10,
    vmax=10,
    set_limits=None,
    save_to_file=None,
    run_names=["run_A", "run_B"],
    figax=None,
):
    """
    This function plots the difference between two variables (var_A, var_B) from the dataset (ds) at
    time (t) on the UFE mesh.
    """

    # Get colormap data
    cmap, norm = get_colormaps()

    ds_A = ds_AA[var].sel(time=t)
    ds_B = ds_BB[var].sel(time=t)

    # Compute difference
    ds_diff = ds_A - ds_B

    # Prepare figure
    if figax == None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    else:
        fig, ax = figax[0], figax[1]

    # Create patch with variable-dependent colormap data
    p = PatchCollection(
        patches, cmap="cmo.balance", norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    )

    # If var = BMB or similar, multiply by -1 to ensure: RED means var(time_A) increased wrsp var(time_B)
    if var in ["BMB", "BMB_inv", "BMB_transition_phase"]:
        cols = [-ds_diff[vi].values for vi in range(0, len(ds_AA.vi))]
    else:
        cols = [ds_diff[vi].values for vi in range(0, len(ds_AA.vi))]

    # Attach data to patch
    p.set_array(cols)

    # Plot cells
    im = ax.add_collection(p)

    # Show colorbar
    fig.colorbar(
        im,
        ax=ax,
        label=f"Difference {run_names[0]} minus {run_names[1]}",
        orientation="vertical",
        shrink=0.7,
        pad=0.05,
        extend="both",
    )

    # Make up grid of subplot
    ax.set_xlim([ds_AA.xmin, ds_AA.xmax])
    ax.set_ylim([ds_AA.ymin, ds_AA.ymax])
    ax.set_aspect(1)
    ax.grid(True)

    # Mask out specified areas
    if "ocean" in mask:
        ds_mask_oc_flat = xr.where(ds_AA.sel(time=t)["mask"] == 2, 2, np.nan)
        p_oc = PatchCollection(patches, cmap=cmap["mask"], norm=norm["mask"])
        cols_mask = [ds_mask_oc_flat[vi].values for vi in range(0, len(ds_mask_oc_flat.vi))]
        p_oc.set_array(cols_mask)
        ax.add_collection(p_oc)

    if "grounded" in mask:
        ds_mask_gr_flat = xr.where(ds_AA.sel(time=t)["mask"] == 3, 3, np.nan)
        p_gr = PatchCollection(patches, cmap=cmap["mask"], norm=norm["mask"])
        cols_mask = [ds_mask_gr_flat[vi].values for vi in range(0, len(ds_mask_gr_flat.vi))]
        p_gr.set_array(cols_mask)
        ax.add_collection(p_gr)

    if "GL_gr" in mask:
        ds_mask_gl_flat = xr.where(ds_AA.sel(time=t)["mask"] == 5, 5, np.nan)
        p_gl = PatchCollection(patches, cmap=cmap["mask"], norm=norm["mask"])
        cols_mask = [ds_mask_gl_flat[vi].values for vi in range(0, len(ds_mask_gl_flat.vi))]
        p_gl.set_array(cols_mask)
        ax.add_collection(p_gl)

    if "GL_fl" in mask:
        ds_mask_gl_flat = xr.where(ds_AA.sel(time=t)["mask"] == 6, 6, np.nan)
        p_gl = PatchCollection(patches, cmap=cmap["mask"], norm=norm["mask"])
        cols_mask = [ds_mask_gl_flat[vi].values for vi in range(0, len(ds_mask_gl_flat.vi))]
        p_gl.set_array(cols_mask)
        ax.add_collection(p_gl)

    # Zoom in on WAIS
    if set_limits == "WAIS":
        ax.set_xlim([-2000000, 0])
        ax.set_ylim([-1500000, 500000])

    elif set_limits == "WAIS_zoom":
        ax.set_xlim([-1750000, -1250000])
        ax.set_ylim([-750000, -200000])

    ax.set_title(f"Difference at time {t}, variable {var} \n {run_names[0]} minus {run_names[1]}")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect(1)
    ax.grid(True, ls="dashed", alpha=0.3)

    fig.tight_layout()

    if save_to_file:
        # Save figure
        plt.savefig(f"{save_to_file}")

    return
