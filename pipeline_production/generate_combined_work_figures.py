#!/usr/bin/env python3
"""Generate the last two figures from notebooks/combined_work_analysis.ipynb."""


from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (16, 6)
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12


BASE_DIR = Path("/home/fbellisardi/code/topolity/data/data_processed")
DEFAULT_OUTPUT_ROOT = Path("/home/fbellisardi/code/topolity/paper/images/figPenultima/test")


def find_result_csvs(city: str, source_filter: str = "graphs"):
    city_dir = BASE_DIR / city
    if not city_dir.exists():
        raise FileNotFoundError(f"City directory not found: {city_dir}")

    csv_paths = []
    if source_filter in ["all", "graphs"]:
        graphs_dir = city_dir / "graphs"
        if graphs_dir.exists():
            csv_paths.extend(list(graphs_dir.glob("*.csv")))

    if source_filter in ["all", "fine_grid"]:
        fine_grid_dir = city_dir / "graphs_fine_grid"
        if fine_grid_dir.exists():
            csv_paths.extend(list(fine_grid_dir.glob("*.csv")))

    if source_filter == "graphs" and not csv_paths:
        fine_grid_dir = city_dir / "graphs_fine_grid"
        if fine_grid_dir.exists():
            csv_paths.extend(list(fine_grid_dir.glob("*.csv")))

    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found for {city} (source_filter={source_filter})")

    return sorted(csv_paths)


def normalize_csv_format(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    df = df.copy()

    if "total_work" not in df.columns and "work_wd" not in df.columns:
        raise ValueError(f"{filename}: missing work columns")

    if "total_work" in df.columns and "variant_type" in df.columns:
        if "work_wd" not in df.columns:
            df["work_wd"] = 0
        if "work_hol" not in df.columns:
            df["work_hol"] = df["total_work"]
        if "variant" not in df.columns:
            def temp_variant(row):
                vtype = str(row.get("variant_type", "")).lower()
                param = row.get("parameter", 0)
                if vtype == "original":
                    return "original"
                if vtype == "rotation":
                    return f"rot_{param:.2f}deg"
                if vtype == "translation":
                    return f"trans_{int(param)}m"
                return f"{vtype}_{param}"
            df["variant"] = df.apply(temp_variant, axis=1)
        return df

    if "total_work" in df.columns and ("work_wd" not in df.columns or "work_hol" not in df.columns):
        if "work_wd" not in df.columns:
            df["work_wd"] = 0
        if "work_hol" not in df.columns:
            df["work_hol"] = df["total_work"]

    required_cols = {"variant", "work_wd", "work_hol"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{filename}: missing columns {required_cols - set(df.columns)}")

    return df


def classify_transformation_type(variant: str) -> str:
    v = str(variant).lower()
    if v == "original":
        return "original"
    if "rot" in v and "rotation" not in v:
        return "rotation"
    if "trans" in v or "translation" in v:
        return "translation"
    if "scale" in v or "scal" in v:
        return "scale"
    return "other"


def load_all_results(city: str, source_filter: str = "graphs", exclude_scale: bool = True) -> pd.DataFrame:
    csv_paths = find_result_csvs(city, source_filter)
    all_data = []

    for csv_path in csv_paths:
        if csv_path.name.startswith("arc"):
            continue
        try:
            df = pd.read_csv(csv_path)
            df = normalize_csv_format(df, csv_path.name)
        except Exception:
            continue

        if "total_work" not in df.columns:
            df["total_work"] = df["work_wd"] + df["work_hol"]

        df = df.drop_duplicates(subset=["variant"], keep="first").reset_index(drop=True)
        df["city"] = city
        df["source_folder"] = csv_path.parent.name
        df["transformation_type"] = df["variant"].apply(classify_transformation_type)
        all_data.append(df)

    if not all_data:
        raise ValueError(f"No valid CSV data for {city}")

    combined = pd.concat(all_data, ignore_index=True)
    if exclude_scale:
        combined = combined[combined["transformation_type"] != "scale"].copy()
    return combined


def plot_combined_work_comparison(results_df: pd.DataFrame, out_png: Path):
    from matplotlib.ticker import ScalarFormatter

    axis_label_size = 24
    tick_label_size = 24

    def format_variant_label(row, trans_counter):
        variant = str(row.get("variant", "unknown"))
        trans_type = row.get("transformation_type", None)

        if variant == "original":
            return "original"

        if trans_type == "rotation":
            parts = variant.split("_")
            angle = None
            for p in parts:
                try:
                    angle = int(p)
                    break
                except Exception:
                    continue
            if angle is not None:
                sign = "+" if angle >= 0 else ""
                return f"rot {sign}{angle}\N{DEGREE SIGN}"
            return variant

        if trans_type == "translation":
            if variant not in trans_counter:
                trans_counter[variant] = len(trans_counter) + 1
            return f"T{trans_counter[variant]}"

        return variant

    df = results_df.copy().drop_duplicates(subset=["variant"], keep="first")
    original = df[df["variant"] == "original"].copy()
    if original.empty:
        raise ValueError("No original row found")

    others_r = df[df["transformation_type"] == "rotation"].sort_values("total_work", ascending=False).copy()
    others_t = df[df["transformation_type"] == "translation"].sort_values("total_work", ascending=False).copy()
    df = pd.concat([original, others_r, others_t], ignore_index=True)

    df = df.sort_values("total_work", ascending=True).reset_index(drop=True)
    block = 3
    orig_row = df[df["variant"] == "original"].iloc[0]
    rest = df[df["variant"] != "original"].reset_index(drop=True)

    def interleave_block(b):
        r = b[b["transformation_type"] == "rotation"].sort_values("total_work", ascending=True)
        t = b[b["transformation_type"] == "translation"].sort_values("total_work", ascending=True)
        out = []
        i = j = 0
        while i < len(r) or j < len(t):
            if i < len(r):
                out.append(r.iloc[i])
                i += 1
            if j < len(t):
                out.append(t.iloc[j])
                j += 1
        return pd.DataFrame(out)

    mixed_parts = [pd.DataFrame([orig_row])]
    for start in range(0, len(rest), block):
        mixed_parts.append(interleave_block(rest.iloc[start:start + block]))
    df = pd.concat(mixed_parts, ignore_index=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#E63946" if v == "original" else "#0400FF" for v in df["variant"]]
    bars = ax.bar(range(len(df)), df["total_work"], color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)

    for i, v in enumerate(df["variant"]):
        if v == "original":
            bars[i].set_edgecolor("darkred")
            bars[i].set_linewidth(2)

    trans_counter = {}
    labels = [str(format_variant_label(row, trans_counter)).replace("_", " ") for _, row in df.iterrows()]
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=tick_label_size)
    ax.set_ylabel("g-Energy Investment (J·m)", fontsize=axis_label_size)

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    sci_formatter = ScalarFormatter(useMathText=True)
    sci_formatter.set_powerlimits((8, 8))
    ax.yaxis.set_major_formatter(sci_formatter)
    ax.tick_params(axis="y", labelsize=tick_label_size)
    ax.yaxis.get_offset_text().set_fontsize(tick_label_size)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    out_pdf = out_png.with_suffix(".pdf")
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_boxplot_by_transformation(results_df: pd.DataFrame, out_png: Path):
    tick_label_size = 24

    df = results_df.copy()
    original_values = df[df["variant"] == "original"]["total_work"].values
    if len(original_values) == 0:
        raise ValueError("No original data found")
    original_value = original_values[0]

    original_df = df[df["transformation_type"] == "original"].copy()
    others_df = df[(df["transformation_type"].isin(["rotation", "translation"])) & (df["total_work"] >= original_value)].copy()
    df_filtered = pd.concat([original_df, others_df], ignore_index=True)
    if df_filtered.empty:
        raise ValueError("No data available for boxplot")

    q99_per_type = df_filtered.groupby("transformation_type")["total_work"].quantile(0.99)
    df_filtered = df_filtered[df_filtered.apply(lambda row: row["total_work"] <= q99_per_type[row["transformation_type"]], axis=1)].copy()

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.boxplot(
        data=df_filtered,
        x="transformation_type",
        y="total_work",
        hue="transformation_type",
        ax=ax,
        order=["original", "rotation", "translation"],
        palette={"original": "#E63946", "rotation": "#fdae61", "translation": "#2b83ba"},
        linewidth=1.5,
        legend=False,
        showfliers=False,
    )

    ax.grid(False)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.tick_params(axis="both", labelsize=tick_label_size)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.ticker import ScalarFormatter
    sci_formatter = ScalarFormatter(useMathText=True)
    sci_formatter.set_powerlimits((8, 8))
    ax.yaxis.set_major_formatter(sci_formatter)
    ax.yaxis.get_offset_text().set_fontsize(tick_label_size)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    out_pdf = out_png.with_suffix(".pdf")
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def discover_step2_cities() -> list:
    cities = []
    for city_dir in sorted(BASE_DIR.iterdir()):
        if not city_dir.is_dir():
            continue
        step2_csv = city_dir / "graphs_fine_grid" / "fine_grid_gravitational_work.csv"
        if step2_csv.exists():
            cities.append(city_dir.name)
    return cities


def main():
    parser = argparse.ArgumentParser(description="Generate combined-work figures for analyzed cities")
    parser.add_argument("--cities", nargs="+", default=None, help="Explicit city list")
    parser.add_argument("--source-filter", choices=["all", "graphs", "fine_grid"], default="graphs")
    parser.add_argument("--include-scale", action="store_true", help="Include scale transformations")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--auto-from-step2", action="store_true", help="Use cities having fine_grid_gravitational_work.csv")
    args = parser.parse_args()

    output_root = Path(args.output_root)

    if args.cities:
        cities = args.cities
    elif args.auto_from_step2:
        cities = discover_step2_cities()
    else:
        cities = discover_step2_cities()

    if not cities:
        raise SystemExit("No cities to process. Provide --cities or run step2 first.")

    print(f"Cities to process: {', '.join(cities)}")
    for city in cities:
        try:
            print(f"\nProcessing city: {city}")
            df = load_all_results(city, source_filter=args.source_filter, exclude_scale=not args.include_scale)
            city_out = output_root / city

            out_combined = city_out / f"{city}_combined_work_comparison.png"
            plot_combined_work_comparison(df, out_combined)
            print(f"  Saved: {out_combined}")

            out_box = city_out / f"{city}_boxplot_by_transformation.png"
            plot_boxplot_by_transformation(df, out_box)
            print(f"  Saved: {out_box}")

        except Exception as e:
            print(f"  [skip] {city}: {e}")


if __name__ == "__main__":
    main()
