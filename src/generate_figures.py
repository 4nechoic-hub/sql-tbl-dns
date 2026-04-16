"""
Publication-quality figures from KTH DNS boundary layer data.

Generates figures showing the full signal processing and turbulence
analysis pipeline using real DNS data from Schlatter & Örlü (2010).
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DatabaseConfig
from src.db import get_engine

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 11, "axes.labelsize": 13, "axes.titlesize": 13,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.linewidth": 0.8,
})

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "data" / "processed"
os.makedirs(OUTPUT, exist_ok=True)

RE_THETAS = [677, 1007, 1421, 2001, 2537, 3032, 3274, 3626, 3969, 4061]
CMAP = plt.cm.viridis
RE_COLOURS = {re: CMAP(i / (len(RE_THETAS) - 1)) for i, re in enumerate(RE_THETAS)}

def _resolve_path(path_like: str) -> str:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path)

def get_engine_db(backend: str = "sqlite", db_path: str | None = None):
    config = DatabaseConfig.from_env(backend=backend)
    if backend == "sqlite":
        config.sqlite_path = _resolve_path(db_path or config.sqlite_path)
    return get_engine(config)

def get_re_colour(re_theta):
    closest = min(RE_THETAS, key=lambda x: abs(x - re_theta))
    return RE_COLOURS[closest]

def fig01_mean_velocity(engine):
    df = pd.read_sql("""
    SELECT sc.re_theta, vp.y_plus, vp.u_plus
    FROM velocity_profiles vp
    JOIN simulation_conditions sc ON vp.condition_id = sc.condition_id
    WHERE vp.y_plus > 0.05 ORDER BY sc.re_theta, vp.y_plus""", engine)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for re in sorted(df["re_theta"].unique()):
        sub = df[df["re_theta"] == re]
        ax.semilogx(sub["y_plus"], sub["u_plus"], color=get_re_colour(re), lw=1.0, label=f"$Re_\\theta = {re:.0f}$")
    yp = np.logspace(1.5, 3.2, 100)
    ax.semilogx(yp, (1/0.41)*np.log(yp)+5.2, "k--", lw=1.5, alpha=0.6, label="Log law ($\\kappa=0.41$, $B=5.2$)")
    yp_v = np.linspace(0.1, 10, 50)
    ax.semilogx(yp_v, yp_v, "k:", lw=1.2, alpha=0.5, label="$U^+ = y^+$")
    ax.set_xlabel("$y^+$"); ax.set_ylabel("$U^+$")
    ax.set_title("Mean Velocity Profiles — Inner Scaling")
    ax.legend(fontsize=7, ncol=2, framealpha=0.9, loc="upper left")
    ax.set_xlim(0.1, 5000); ax.set_ylim(0, 32)
    plt.tight_layout(); fig.savefig(f"{OUTPUT}/fig01_mean_velocity.png"); plt.close(fig)
    print("  [01] Mean velocity profiles")

def fig02_reynolds_stresses(engine):
    df = pd.read_sql("""
    SELECT sc.re_theta, vp.y_plus, vp.u_rms_plus, vp.v_rms_plus, vp.w_rms_plus, vp.uv_plus
    FROM velocity_profiles vp JOIN simulation_conditions sc ON vp.condition_id = sc.condition_id
    WHERE vp.y_plus > 0.05 ORDER BY sc.re_theta, vp.y_plus""", engine)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for col, label, ax in [("u_rms_plus","$u'^+_{rms}$",axes[0,0]),("v_rms_plus","$v'^+_{rms}$",axes[0,1]),
                            ("w_rms_plus","$w'^+_{rms}$",axes[1,0]),("uv_plus","$\\langle u'v' \\rangle^+$",axes[1,1])]:
        for re in sorted(df["re_theta"].unique()):
            sub = df[df["re_theta"]==re]
            ax.semilogx(sub["y_plus"], sub[col], color=get_re_colour(re), lw=0.8, label=f"{re:.0f}")
        ax.set_xlabel("$y^+$"); ax.set_ylabel(label); ax.set_xlim(0.5, 5000)
    axes[0,0].set_title("(a) Streamwise RMS"); axes[0,1].set_title("(b) Wall-normal RMS")
    axes[1,0].set_title("(c) Spanwise RMS"); axes[1,1].set_title("(d) Reynolds Shear Stress")
    h, l = axes[0,0].get_legend_handles_labels()
    fig.legend(h, [f"$Re_\\theta={x}$" for x in l], loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5,-0.02))
    plt.suptitle("Reynolds Stress Profiles — Inner Scaling", fontsize=14, y=1.01)
    plt.tight_layout(rect=[0,0.04,1,0.98]); fig.savefig(f"{OUTPUT}/fig02_reynolds_stresses.png"); plt.close(fig)
    print("  [02] Reynolds stress profiles")

def fig03_tke_profiles(engine):
    df = pd.read_sql("""
    SELECT sc.re_theta, vp.y_plus, vp.y_delta,
           0.5*(vp.u_rms_plus*vp.u_rms_plus+vp.v_rms_plus*vp.v_rms_plus+vp.w_rms_plus*vp.w_rms_plus) AS tke_plus
    FROM velocity_profiles vp JOIN simulation_conditions sc ON vp.condition_id=sc.condition_id
    WHERE vp.y_plus > 0.05 ORDER BY sc.re_theta, vp.y_plus""", engine)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for re in sorted(df["re_theta"].unique()):
        sub = df[df["re_theta"]==re]
        axes[0].semilogx(sub["y_plus"], sub["tke_plus"], color=get_re_colour(re), lw=1.0, label=f"{re:.0f}")
        axes[1].plot(sub["y_delta"], sub["tke_plus"], color=get_re_colour(re), lw=1.0)
    axes[0].set_xlabel("$y^+$"); axes[0].set_ylabel("$k^+$"); axes[0].set_title("(a) Inner Scaling"); axes[0].set_xlim(0.5,5000)
    axes[1].set_xlabel("$y/\\delta_{99}$"); axes[1].set_ylabel("$k^+$"); axes[1].set_title("(b) Outer Scaling"); axes[1].set_xlim(0,1.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, [f"$Re_\\theta={x}$" for x in l], loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5,-0.02))
    plt.suptitle("Turbulent Kinetic Energy Profiles", fontsize=14, y=1.01)
    plt.tight_layout(rect=[0,0.04,1,0.98]); fig.savefig(f"{OUTPUT}/fig03_tke_profiles.png"); plt.close(fig)
    print("  [03] TKE profiles")

def fig04_tke_budget(engine):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, re_target, panel in zip(axes, [677, 4061], ["(a)", "(b)"]):
        df = pd.read_sql(f"""
        SELECT tb.y_plus, tb.production, tb.dissipation, tb.turb_diffusion, tb.vel_pressure, tb.visc_diffusion
        FROM tke_budgets tb JOIN simulation_conditions sc ON tb.condition_id=sc.condition_id
        WHERE sc.re_theta=(SELECT re_theta FROM simulation_conditions ORDER BY ABS(re_theta-{re_target}) LIMIT 1)
        AND tb.y_plus>0.05 AND tb.y_plus<300 ORDER BY tb.y_plus""", engine)
        re_actual = pd.read_sql(f"SELECT re_theta FROM simulation_conditions ORDER BY ABS(re_theta-{re_target}) LIMIT 1", engine)["re_theta"].iloc[0]
        ax.semilogx(df["y_plus"], df["production"], "b-", lw=1.3, label="Production")
        ax.semilogx(df["y_plus"], df["dissipation"], "r-", lw=1.3, label="Dissipation")
        ax.semilogx(df["y_plus"], df["turb_diffusion"], "g--", lw=1.1, label="Turb. diff.")
        ax.semilogx(df["y_plus"], df["vel_pressure"], "m--", lw=1.1, label="Vel.-press.")
        ax.semilogx(df["y_plus"], df["visc_diffusion"], "c--", lw=1.1, label="Visc. diff.")
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax.set_xlabel("$y^+$"); ax.set_ylabel("Budget terms$^+$")
        ax.set_title(f"{panel} $Re_\\theta \\approx {re_actual:.0f}$"); ax.legend(fontsize=8); ax.set_xlim(0.5, 300)
    plt.suptitle("TKE Budget — Production, Dissipation, and Transport", fontsize=14, y=1.01)
    plt.tight_layout(); fig.savefig(f"{OUTPUT}/fig04_tke_budget.png"); plt.close(fig)
    print("  [04] TKE budget")

def fig05_integral_params(engine):
    df = pd.read_sql("SELECT re_theta, re_tau, cf, shape_factor FROM simulation_conditions ORDER BY re_theta", engine)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].plot(df["re_theta"], df["cf"]*1000, "ko-", ms=6, lw=1.2)
    axes[0].set_xlabel("$Re_\\theta$"); axes[0].set_ylabel("$c_f \\times 10^3$"); axes[0].set_title("(a) Skin Friction")
    axes[1].plot(df["re_theta"], df["re_tau"], "ko-", ms=6, lw=1.2)
    axes[1].set_xlabel("$Re_\\theta$"); axes[1].set_ylabel("$Re_\\tau$"); axes[1].set_title("(b) Friction Reynolds Number")
    axes[2].plot(df["re_theta"], df["shape_factor"], "ko-", ms=6, lw=1.2)
    axes[2].axhline(1.4, color="r", ls=":", alpha=0.5); axes[2].set_xlabel("$Re_\\theta$")
    axes[2].set_ylabel("$H_{12}$"); axes[2].set_title("(c) Shape Factor")
    plt.suptitle("Integral Boundary Layer Parameters vs $Re_\\theta$", fontsize=14, y=1.02)
    plt.tight_layout(); fig.savefig(f"{OUTPUT}/fig05_integral_params.png"); plt.close(fig)
    print("  [05] Integral parameters")

def fig06_contour_map(engine):
    df = pd.read_sql("""
    SELECT sc.re_theta, vp.y_plus, vp.u_rms_plus,
           0.5*(vp.u_rms_plus*vp.u_rms_plus+vp.v_rms_plus*vp.v_rms_plus+vp.w_rms_plus*vp.w_rms_plus) AS tke_plus
    FROM velocity_profiles vp JOIN simulation_conditions sc ON vp.condition_id=sc.condition_id
    WHERE vp.y_plus BETWEEN 0.5 AND 1500 ORDER BY sc.re_theta, vp.y_plus""", engine)
    re_vals = sorted(df["re_theta"].unique())
    yp_common = np.logspace(np.log10(0.5), np.log10(1500), 200)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for col, cbl, cm, ax, title in [("u_rms_plus","$u'^+_{rms}$","YlOrRd",axes[0],"(a) Streamwise RMS"),
                                     ("tke_plus","$k^+$","inferno",axes[1],"(b) TKE")]:
        Z = np.zeros((len(yp_common), len(re_vals)))
        for j, re in enumerate(re_vals):
            sub = df[df["re_theta"]==re].sort_values("y_plus")
            Z[:, j] = np.interp(yp_common, sub["y_plus"].values, sub[col].values)
        RE, YP = np.meshgrid(re_vals, yp_common)
        cf = ax.contourf(RE, YP, Z, levels=25, cmap=cm)
        ax.contour(RE, YP, Z, levels=10, colors="k", linewidths=0.2, alpha=0.3)
        plt.colorbar(cf, ax=ax, label=cbl, pad=0.02)
        ax.set_xlabel("$Re_\\theta$"); ax.set_ylabel("$y^+$"); ax.set_yscale("log"); ax.set_title(title)
    plt.suptitle("Turbulence Field — $Re_\\theta$ vs $y^+$ Contour Maps", fontsize=14, y=1.01)
    plt.tight_layout(); fig.savefig(f"{OUTPUT}/fig06_contour_maps.png"); plt.close(fig)
    print("  [06] Contour maps")

def fig07_higher_order(engine):
    df = pd.read_sql("""
    SELECT sc.re_theta, vp.y_plus, vp.skewness_u, vp.flatness_u
    FROM velocity_profiles vp JOIN simulation_conditions sc ON vp.condition_id=sc.condition_id
    WHERE vp.y_plus>0.1 ORDER BY sc.re_theta, vp.y_plus""", engine)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for re in sorted(df["re_theta"].unique()):
        sub = df[df["re_theta"]==re]
        axes[0].semilogx(sub["y_plus"], sub["skewness_u"], color=get_re_colour(re), lw=0.8, label=f"{re:.0f}")
        axes[1].semilogx(sub["y_plus"], sub["flatness_u"], color=get_re_colour(re), lw=0.8)
    axes[0].axhline(0, color="k", lw=0.5, alpha=0.3)
    axes[0].set_xlabel("$y^+$"); axes[0].set_ylabel("$S(u)$"); axes[0].set_title("(a) Skewness"); axes[0].set_xlim(0.1,5000)
    axes[1].axhline(3, color="k", ls=":", lw=0.8, alpha=0.5)
    axes[1].text(3000, 3.15, "Gaussian", fontsize=8, color="0.4")
    axes[1].set_xlabel("$y^+$"); axes[1].set_ylabel("$F(u)$"); axes[1].set_title("(b) Flatness"); axes[1].set_xlim(0.1,5000)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, [f"$Re_\\theta={x}$" for x in l], loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5,-0.02))
    plt.suptitle("Higher-Order Statistics", fontsize=14, y=1.01)
    plt.tight_layout(rect=[0,0.04,1,0.98]); fig.savefig(f"{OUTPUT}/fig07_higher_order.png"); plt.close(fig)
    print("  [07] Higher-order statistics")

def fig08_defect_profile(engine):
    df = pd.read_sql("""
    SELECT sc.re_theta, vp.y_delta, vp.u_plus
    FROM velocity_profiles vp JOIN simulation_conditions sc ON vp.condition_id=sc.condition_id
    WHERE vp.y_delta BETWEEN 0.01 AND 1.5 ORDER BY sc.re_theta, vp.y_delta""", engine)
    fig, ax = plt.subplots(figsize=(8, 6))
    for re in sorted(df["re_theta"].unique()):
        sub = df[df["re_theta"]==re]
        u_inf_plus = sub["u_plus"].max()
        ax.semilogy(sub["y_delta"], u_inf_plus-sub["u_plus"], color=get_re_colour(re), lw=1.0, label=f"$Re_\\theta={re:.0f}$")
    ax.set_xlabel("$y / \\delta_{99}$"); ax.set_ylabel("$U^+_\\infty - U^+$")
    ax.set_title("Velocity Defect — Outer Scaling"); ax.set_xlim(0, 1.3)
    ax.legend(fontsize=7, ncol=2, framealpha=0.9)
    plt.tight_layout(); fig.savefig(f"{OUTPUT}/fig08_defect_profile.png"); plt.close(fig)
    print("  [08] Defect profile")

def fig09_vorticity(engine):
    df = pd.read_sql("""
    SELECT sc.re_theta, vp.y_plus, vp.omx_rms_plus, vp.omy_rms_plus, vp.omz_rms_plus
    FROM velocity_profiles vp JOIN simulation_conditions sc ON vp.condition_id=sc.condition_id
    WHERE vp.y_plus>0.1 ORDER BY sc.re_theta, vp.y_plus""", engine)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for col, yl, ax, t in [("omx_rms_plus","$\\omega'^+_{x}$",axes[0],"(a) Streamwise"),
                            ("omy_rms_plus","$\\omega'^+_{y}$",axes[1],"(b) Wall-normal"),
                            ("omz_rms_plus","$\\omega'^+_{z}$",axes[2],"(c) Spanwise")]:
        for re in sorted(df["re_theta"].unique()):
            sub=df[df["re_theta"]==re]
            ax.semilogx(sub["y_plus"], sub[col], color=get_re_colour(re), lw=0.8)
        ax.set_xlabel("$y^+$"); ax.set_ylabel(yl); ax.set_title(t); ax.set_xlim(0.1,5000)
    h=[plt.Line2D([0],[0],color=get_re_colour(re),lw=1.2) for re in sorted(df["re_theta"].unique())]
    fig.legend(h,[f"$Re_\\theta={re:.0f}$" for re in sorted(df["re_theta"].unique())],
              loc="lower center",ncol=5,fontsize=8,bbox_to_anchor=(0.5,-0.02))
    plt.suptitle("Vorticity RMS Profiles", fontsize=14, y=1.01)
    plt.tight_layout(rect=[0,0.04,1,0.98]); fig.savefig(f"{OUTPUT}/fig09_vorticity.png"); plt.close(fig)
    print("  [09] Vorticity profiles")

def fig10_3d_surface(engine):
    df = pd.read_sql("""
    SELECT sc.re_theta, vp.y_plus, vp.u_rms_plus
    FROM velocity_profiles vp JOIN simulation_conditions sc ON vp.condition_id=sc.condition_id
    WHERE vp.y_plus BETWEEN 0.5 AND 500 ORDER BY sc.re_theta, vp.y_plus""", engine)
    re_vals = sorted(df["re_theta"].unique())
    yp_common = np.logspace(np.log10(0.5), np.log10(500), 150)
    Z = np.zeros((len(yp_common), len(re_vals)))
    for j, re in enumerate(re_vals):
        sub=df[df["re_theta"]==re].sort_values("y_plus")
        Z[:,j]=np.interp(yp_common, sub["y_plus"].values, sub["u_rms_plus"].values)
    RE, YP = np.meshgrid(re_vals, yp_common)
    fig=plt.figure(figsize=(11,7))
    ax=fig.add_subplot(111, projection="3d")
    surf=ax.plot_surface(RE/1000, np.log10(YP), Z, cmap="coolwarm", alpha=0.9,
                         edgecolor="k", linewidth=0.05, rstride=3, cstride=1)
    ax.set_xlabel("\n$Re_\\theta \\times 10^{-3}$", fontsize=11, labelpad=12)
    ax.set_ylabel("\n$\\log_{10}(y^+)$", fontsize=11, labelpad=12)
    ax.set_zlabel("\n$u'^+_{rms}$", fontsize=11, labelpad=8)
    ax.set_title("Streamwise Turbulence Intensity", fontsize=13, pad=12)
    ax.view_init(elev=25, azim=-50)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.1, label="$u'^+_{rms}$")
    plt.tight_layout(); fig.savefig(f"{OUTPUT}/fig10_3d_surface.png"); plt.close(fig)
    print("  [10] 3D surface")

def main():
    parser = argparse.ArgumentParser(description="Generate publication-quality figures from KTH DNS data")
    parser.add_argument("--backend", choices=["sqlite", "postgresql"], default="sqlite")
    parser.add_argument(
        "--db-path",
        default=None,
        help="SQLite database path override (used only when --backend sqlite)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data" / "processed"),
        help="Directory where PNG figures will be written",
    )
    args = parser.parse_args()

    if args.backend != "sqlite" and args.db_path is not None:
        parser.error("--db-path is only valid when --backend sqlite")

    global OUTPUT
    OUTPUT = _resolve_path(args.output_dir)
    os.makedirs(OUTPUT, exist_ok=True)

    print(f"Generating figures from KTH DNS data using {args.backend}...")
    engine = get_engine_db(backend=args.backend, db_path=args.db_path)
    fig01_mean_velocity(engine)
    fig02_reynolds_stresses(engine)
    fig03_tke_profiles(engine)
    fig04_tke_budget(engine)
    fig05_integral_params(engine)
    fig06_contour_map(engine)
    fig07_higher_order(engine)
    fig08_defect_profile(engine)
    fig09_vorticity(engine)
    fig10_3d_surface(engine)
    print(f"\nAll figures saved to {OUTPUT}/")
    for f in sorted(os.listdir(OUTPUT)):
        if f.startswith("fig") and f.endswith(".png"):
            print(f"  {f:40s} {os.path.getsize(f'{OUTPUT}/{f}')/1024:6.0f} KB")


if __name__ == "__main__":
    main()
