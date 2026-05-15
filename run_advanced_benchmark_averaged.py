#!/usr/bin/env python3
# run_advanced_benchmark_averaged.py
#
# Делает больше тестов и усредняет результаты по нескольким seed.
#
# Использует файл:
#   centroid_benchmark_advanced_verbose.cpp
#
# Что делает:
#   1. Патчит C++ так, чтобы seed передавался аргументом командной строки.
#   2. Запускает benchmark несколько раз с разными seed.
#   3. Склеивает результаты.
#   4. Считает mean и median по каждой точке.
#   5. Строит графики времени по median (без графиков «ускорения»).
#
# Подробности: README.md
#
# Запуск:
#   python run_advanced_benchmark_averaged.py quick 3
#   python run_advanced_benchmark_averaged.py replot quick
#
# quick/demo: random+path, n=300..30000 шаг 100, q=50000, ~1788 кейсов/seed

import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent

CPP_SOURCE = ROOT / "centroid_benchmark_advanced_verbose.cpp"
CPP_PATCHED = ROOT / "centroid_benchmark_averaged.cpp"
EXE_FILE = ROOT / "centroid_benchmark_averaged.exe"

RAW_CSV = ROOT / "results_averaged_raw.csv"
AGG_CSV = ROOT / "results_averaged_median.csv"

SUMMARY_TXT = ROOT / "averaged_benchmark_summary.txt"


def summary_chart_paths(mode: str) -> dict[str, Path]:
    prefix = "presentation" if mode == "presentation" else "averaged"
    return {
        "time": ROOT / f"{prefix}_benchmark_time.png",
        "time_total": ROOT / f"{prefix}_benchmark_time_total.png",
        "memory": ROOT / f"{prefix}_benchmark_memory.png",
        "time_random": ROOT / f"{prefix}_benchmark_time_random.png",
        "time_growth": ROOT / f"{prefix}_benchmark_time_growth.png",
    }


TREE_ORDER = ["random", "path"]
TREE_LABELS = {
    "random": "случайное дерево",
    "path": "путь",
}


def workload_for_mode(mode: str) -> str:
    return "deck_50_50" if mode == "presentation" else "balanced"


def output_paths(mode: str) -> dict:
    if mode == "presentation":
        return {
            "raw": ROOT / "results_presentation_raw.csv",
            "agg": ROOT / "results_presentation_median.csv",
            "summary": ROOT / "presentation_benchmark_summary.txt",
        }
    return {
        "raw": RAW_CSV,
        "agg": AGG_CSV,
        "summary": SUMMARY_TXT,
    }


def patch_cpp():
    if not CPP_SOURCE.exists():
        raise FileNotFoundError(
            f"Не найден {CPP_SOURCE.name}. Положи этот скрипт рядом с centroid_benchmark_advanced_verbose.cpp"
        )

    source = CPP_SOURCE.read_text(encoding="utf-8")

    # Добавляем seed из argv[2].
    source = source.replace(
        "    mt19937 rng(42);\n",
        """    int seed = 42;
    if (argc >= 3) seed = atoi(argv[2]);
    mt19937 rng(seed);
    cerr << "Seed: " << seed << "\\n";
"""
    )

    CPP_PATCHED.write_text(source, encoding="utf-8")


def compile_cpp():
    cmd = [
        "g++",
        "-O2",
        "-std=c++17",
        str(CPP_PATCHED),
        "-o",
        str(EXE_FILE),
    ]
    if sys.platform == "win32":
        cmd.extend(["-lpsapi"])

    print("$", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def run_one(mode: str, seed: int, run_index: int, repeats: int) -> pd.DataFrame:
    tmp_csv = ROOT / f"results_tmp_seed_{seed}.csv"

    print()
    print(f"=== Повтор {run_index}/{repeats}, seed={seed}, mode={mode} ===")

    with tmp_csv.open("w", encoding="utf-8", newline="") as f:
        subprocess.run(
            [str(EXE_FILE), mode, str(seed)],
            check=True,
            stdout=f,
        )

    df = pd.read_csv(tmp_csv)
    df.insert(0, "seed", seed)
    tmp_csv.unlink(missing_ok=True)
    return df


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["tree_type", "workload", "n", "queries"]

    metric_cols = [
        "paint_count",
        "query_count",
        "centroid_build_ms",
        "centroid_run_ms",
        "naive_bfs_run_ms",
        "lca_build_ms",
        "lca_run_ms",
        "speedup_vs_naive",
        "speedup_vs_lca",
        "centroid_memory_bytes",
        "naive_memory_bytes",
        "lca_memory_bytes",
    ]

    # Для графиков лучше median: она устойчивее к выбросам.
    med = raw.groupby(group_cols, as_index=False)[metric_cols].median()

    # Дополнительно сохраняем mean и std для проверки разброса.
    mean = raw.groupby(group_cols, as_index=False)[metric_cols].mean()
    std = raw.groupby(group_cols, as_index=False)[metric_cols].std().fillna(0)

    # Добавим несколько колонок mean/std для speedup, чтобы понимать шум.
    med = med.merge(
        mean[group_cols + ["speedup_vs_naive", "centroid_run_ms", "naive_bfs_run_ms"]]
            .rename(columns={
                "speedup_vs_naive": "speedup_vs_naive_mean",
                "centroid_run_ms": "centroid_run_ms_mean",
                "naive_bfs_run_ms": "naive_bfs_run_ms_mean",
            }),
        on=group_cols,
        how="left",
    )

    med = med.merge(
        std[group_cols + ["speedup_vs_naive", "centroid_run_ms", "naive_bfs_run_ms"]]
            .rename(columns={
                "speedup_vs_naive": "speedup_vs_naive_std",
                "centroid_run_ms": "centroid_run_ms_std",
                "naive_bfs_run_ms": "naive_bfs_run_ms_std",
            }),
        on=group_cols,
        how="left",
    )

    return med


def _filter_tree(df: pd.DataFrame, tree_type: str, workload: str) -> pd.DataFrame:
    part = df[(df["workload"] == workload) & (df["tree_type"] == tree_type)].copy()
    return part.sort_values("n")


def _n_values(part: pd.DataFrame) -> list[int]:
    return sorted(int(x) for x in part["n"].unique())


def _with_per_query_metrics(part: pd.DataFrame) -> pd.DataFrame:
    out = part.copy()
    q = out["query_count"].clip(lower=1)
    out["centroid_us_per_query"] = out["centroid_run_ms"] * 1000.0 / q
    out["naive_us_per_query"] = out["naive_bfs_run_ms"] * 1000.0 / q
    out["sqrt_lca_us_per_query"] = out["lca_run_ms"] * 1000.0 / q
    return out


def _with_total_time_metrics(part: pd.DataFrame) -> pd.DataFrame:
    out = part.copy()
    out["centroid_total_ms"] = out["centroid_build_ms"] + out["centroid_run_ms"]
    out["sqrt_lca_total_ms"] = out["lca_build_ms"] + out["lca_run_ms"]
    out["naive_total_ms"] = out["naive_bfs_run_ms"]
    return out


def _configure_log_n_axis(ax, n_vals: list[int]) -> None:
    if not n_vals:
        return
    ax.set_xscale("log")
    if len(n_vals) > 18:
        from matplotlib.ticker import LogLocator, LogFormatterSciNotation

        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
        ax.xaxis.set_major_formatter(LogFormatterSciNotation())
    else:
        ax.set_xticks(n_vals)
        ax.set_xticklabels([str(v) for v in n_vals], rotation=45, ha="right", fontsize=7)
    ax.minorticks_on()


LINE_STYLES = {
    "centroid": {"color": "C0", "linestyle": "-", "linewidth": 1.8},
    "sqrt_lca": {"color": "C1", "linestyle": "-.", "linewidth": 1.8},
    "naive": {"color": "C2", "linestyle": "--", "linewidth": 1.8},
}


def _plot_three_methods(
    ax,
    part: pd.DataFrame,
    title: str,
    ylabel: str,
    centroid_col: str,
    sqrt_lca_col: str,
    naive_col: str,
    log_y: bool = False,
) -> None:
    ax.plot(part["n"], part[centroid_col], label="Centroid", alpha=0.92, **LINE_STYLES["centroid"])
    ax.plot(
        part["n"],
        part[sqrt_lca_col],
        label="sqrt(n)-батчи + LCA",
        alpha=0.92,
        **LINE_STYLES["sqrt_lca"],
    )
    ax.plot(part["n"], part[naive_col], label="Naive BFS", alpha=0.92, **LINE_STYLES["naive"])
    _configure_log_n_axis(ax, _n_values(part))
    if log_y:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("n (вершин)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(fontsize=8)


def _reference_curves(n: np.ndarray, y0: float, n0: float) -> tuple[np.ndarray, np.ndarray]:
    """Масштабные ориентиры O(n) и O(n log n) через первую точку centroid."""
    lin = y0 * (n / n0)
    nlog = y0 * (n * np.log2(np.maximum(n, 2))) / (n0 * np.log2(max(n0, 2)))
    return lin, nlog


def _remove_legacy_per_tree_png(mode: str) -> None:
    prefix = "presentation" if mode == "presentation" else "averaged"
    keep = {p.name for p in summary_chart_paths(mode).values()}
    for path in ROOT.glob(f"{prefix}_*.png"):
        if path.name not in keep:
            path.unlink(missing_ok=True)
            print(f"Удалён устаревший график: {path.name}")


def plot_summary_charts(
    agg: pd.DataFrame,
    mode: str,
    workload: str,
) -> list[Path]:
    slide14 = mode == "presentation"
    paths = summary_chart_paths(mode)
    trees = [t for t in TREE_ORDER if t in agg.loc[agg["workload"] == workload, "tree_type"].unique()]
    if not trees:
        trees = sorted(agg.loc[agg["workload"] == workload, "tree_type"].unique())

    wl_df = agg[agg["workload"] == workload]
    q = int(wl_df["queries"].iloc[0]) if len(wl_df) else 0
    n_count = wl_df["n"].nunique()

    # --- 1. Время: random + path, 3 метода ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    fig.suptitle(
        f"Время на запрос · {n_count} точек n · q={q} · workload={workload}",
        fontsize=12,
    )
    for ax, tree in zip(axes, trees):
        part = _with_per_query_metrics(_filter_tree(agg, tree, workload))
        _plot_three_methods(
            ax,
            part,
            TREE_LABELS.get(tree, tree),
            "мкс / query",
            "centroid_us_per_query",
            "sqrt_lca_us_per_query",
            "naive_us_per_query",
        )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(paths["time"], dpi=220)
    plt.close(fig)
    print(f"График: {paths['time']}")

    # --- 2. Общее время на весь прогон q операций (build + run где есть) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        f"Общее время на прогон · {n_count} точек n · q={q} · workload={workload}",
        fontsize=12,
    )
    for ax, tree in zip(axes, trees):
        part = _with_total_time_metrics(_filter_tree(agg, tree, workload))
        _plot_three_methods(
            ax,
            part,
            TREE_LABELS.get(tree, tree),
            "мс (median)",
            "centroid_total_ms",
            "sqrt_lca_total_ms",
            "naive_total_ms",
            log_y=True,
        )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(paths["time_total"], dpi=220)
    plt.close(fig)
    print(f"График: {paths['time_total']}")

    # --- 3. Память ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    fig.suptitle(f"Память (Private Usage), KiB · workload={workload}", fontsize=12)
    for ax, tree in zip(axes, trees):
        part = _filter_tree(agg, tree, workload)
        ax.plot(
            part["n"],
            part["centroid_memory_bytes"] / 1024,
            label="Centroid",
            alpha=0.92,
            **LINE_STYLES["centroid"],
        )
        ax.plot(
            part["n"],
            part["lca_memory_bytes"] / 1024,
            label="sqrt(n)-батчи + LCA",
            alpha=0.92,
            **LINE_STYLES["sqrt_lca"],
        )
        ax.plot(
            part["n"],
            part["naive_memory_bytes"] / 1024,
            label="Naive BFS",
            alpha=0.92,
            **LINE_STYLES["naive"],
        )
        _configure_log_n_axis(ax, _n_values(part))
        ax.set_yscale("log")
        ax.set_title(TREE_LABELS.get(tree, tree), fontsize=10)
        ax.set_xlabel("n")
        ax.set_ylabel("KiB")
        ax.grid(True, which="both", alpha=0.35)
        ax.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(paths["memory"], dpi=220)
    plt.close(fig)
    print(f"График: {paths['memory']}")

    # --- 4. Random + ориентиры роста ---
    if "random" in trees:
        part = _with_per_query_metrics(_filter_tree(agg, "random", workload))
        n = part["n"].to_numpy(dtype=float)
        y_cd = part["centroid_us_per_query"].to_numpy(dtype=float)
        y_sq = part["sqrt_lca_us_per_query"].to_numpy(dtype=float)
        y_nv = part["naive_us_per_query"].to_numpy(dtype=float)
        ref_lin, ref_nlog = _reference_curves(n, y_cd[0], n[0])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(n, y_cd, label="Centroid", alpha=0.92, **LINE_STYLES["centroid"])
        ax.plot(n, y_sq, label="sqrt(n)-батчи + LCA", alpha=0.92, **LINE_STYLES["sqrt_lca"])
        ax.plot(n, y_nv, label="Naive BFS", alpha=0.92, **LINE_STYLES["naive"])
        ax.plot(n, ref_lin, ":", color="gray", linewidth=1.2, label="ориентир ∝ n")
        ax.plot(n, ref_nlog, ":", color="silver", linewidth=1.2, label="ориентир ∝ n log n")
        _configure_log_n_axis(ax, _n_values(part))
        ax.set_title(f"Случайное дерево · q={q} · workload={workload}", fontsize=11)
        ax.set_xlabel("n")
        ax.set_ylabel("мкс / query (median)")
        ax.grid(True, which="both", alpha=0.35)
        ax.legend()
        fig.tight_layout()
        fig.savefig(paths["time_random"], dpi=220)
        plt.close(fig)
        print(f"График: {paths['time_random']}")

    # --- 5. log-log random ---
    if "random" in trees:
        part = _with_per_query_metrics(_filter_tree(agg, "random", workload))
        n = part["n"].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.loglog(
            n, part["centroid_us_per_query"], label="Centroid", alpha=0.92, **LINE_STYLES["centroid"]
        )
        ax.loglog(
            n,
            part["sqrt_lca_us_per_query"],
            label="sqrt(n)-батчи + LCA",
            alpha=0.92,
            **LINE_STYLES["sqrt_lca"],
        )
        ax.loglog(
            n, part["naive_us_per_query"], label="Naive BFS", alpha=0.92, **LINE_STYLES["naive"]
        )
        ax.set_title("Случайное дерево: log(время) vs log(n)", fontsize=11)
        ax.set_xlabel("n")
        ax.set_ylabel("мкс / query (median)")
        ax.grid(True, which="both", alpha=0.35)
        ax.legend()
        fig.tight_layout()
        fig.savefig(paths["time_growth"], dpi=220)
        plt.close(fig)
        print(f"График: {paths['time_growth']}")

    _remove_legacy_per_tree_png(mode)
    return list(paths.values())


def _n_grid_line(agg: pd.DataFrame, plot_workload: str) -> str:
    n_vals = sorted(int(x) for x in agg["n"].unique())
    trees = sorted(agg["tree_type"].unique())
    workloads = sorted(agg["workload"].unique())
    vals = ", ".join(str(x) for x in n_vals)
    plot_rows = agg[agg["workload"] == plot_workload]
    return (
        f"Сетка n ({len(n_vals)} значений): {vals}\n"
        f"В CSV: {len(agg)} кейсов = {len(trees)} деревьев x {len(workloads)} нагрузок x {len(n_vals)} n\n"
        f"На графиках PNG: workload={plot_workload} "
        f"({len(plot_rows)} строк = {len(trees)} x {len(n_vals)} точек)"
    )


def make_summary(
    raw: pd.DataFrame,
    agg: pd.DataFrame,
    repeats: int,
    mode: str,
    workload: str,
    plot_files: list[Path],
    summary_path: Path,
):
    lines = []
    if mode == "presentation":
        lines.append("Бенчмарк по постановке слайдов 14–15 презентации")
        lines.append("(nearest active / красные вершины: random Prüfer-like, n=10³..10⁵, q=10⁵, p(paint)=0.5)")
        lines.append("")
        lines.append(
            "Слайд 14: в фокусе сравнения Naive (BFS от x до ближайшей активной) и Centroid "
            "(best[c] + цепочки к центроидам)."
        )
        lines.append(
            "«LCA-only» как отдельное решение nearest active не бенчмаркается: без дополнительной "
            "структуры глобальный минимум по красным так не получить (как на слайде)."
        )
        lines.append(
            "В CSV также sqrt+LCA; графики presentation — только Naive vs Centroid."
        )
        lines.append(
            "Память в CSV: прирост Private Usage процесса (Windows) на прогон алгоритма."
        )
    else:
        lines.append("Выводы по усреднённому benchmark")
    lines.append("=" * 36)
    lines.append("")
    lines.append(f"Повторов на точку: {repeats}")
    lines.append(f"Диапазон n: {int(agg['n'].min())} .. {int(agg['n'].max())}")
    lines.append(f"Сырых строк: {len(raw)}")
    lines.append(f"Усреднённых точек: {len(agg)}")
    lines.append("")
    lines.append("Для графиков используется median, а не mean: это уменьшает влияние случайных выбросов.")
    lines.append("")
    lines.append(_n_grid_line(agg, workload))
    lines.append("")
    lines.append("Графики (5 сводных PNG):")
    for p in plot_files:
        lines.append(f"  - {p.name}")
    lines.append("")
    lines.append(
        "Время на запрос: мкс/query (median). Общее время: build+run для Centroid и sqrt+LCA, "
        "только run для Naive, на все q операций; ось Y log."
    )
    lines.append("")

    if mode == "presentation":
        lines.append(
            "Слайд 15: при больших n время наивного BFS растёт сильнее, чем у центроида."
        )
    else:
        lines.append(
            "Три метода: Centroid O(log n), sqrt(n)-батчи+LCA (амортиз. ~sqrt(n)), Naive O(n) на query. "
            "Деревья: только random (Prüfer) и path."
        )

    text = "\n".join(lines)
    summary_path.write_text(text, encoding="utf-8")

    print()
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("cp1251", errors="replace").decode("cp1251"))
    print(f"\nВыводы сохранены в {summary_path}")


def main():
    mode = "quick"
    repeats = 5
    replot_only = False

    if len(sys.argv) >= 2 and sys.argv[1] == "replot":
        replot_only = True
        if len(sys.argv) < 3:
            raise ValueError("Использование: python run_advanced_benchmark_averaged.py replot <demo|quick|...>")
        mode = sys.argv[2]
    elif len(sys.argv) >= 2:
        mode = sys.argv[1]

    if len(sys.argv) >= 3 and not replot_only:
        repeats = int(sys.argv[2])
    elif replot_only and len(sys.argv) >= 4:
        repeats = int(sys.argv[3])

    if mode not in {"smoke", "demo", "quick", "full", "presentation"}:
        raise ValueError("Режим должен быть presentation, smoke, demo, quick или full")

    if repeats < 1:
        raise ValueError("Число повторов должно быть >= 1")

    if mode == "presentation" and repeats > 5:
        print(
            "Предупреждение: режим presentation с большим числом повторов и наивным BFS "
            "на n≈10^5 может считаться очень долго; обычно достаточно 1–3 повторов.",
            file=sys.stderr,
        )

    paths = output_paths(mode)
    wl = workload_for_mode(mode)

    if replot_only:
        if not paths["agg"].exists():
            raise FileNotFoundError(
                f"Нет {paths['agg'].name}. Сначала запусти: python {Path(__file__).name} {mode} 1"
            )
        agg = pd.read_csv(paths["agg"])
        raw = pd.read_csv(paths["raw"]) if paths["raw"].exists() else agg
        print(f"Replot из {paths['agg']}")
    else:
        patch_cpp()
        compile_cpp()

        seeds = [42 + 1009 * i for i in range(repeats)]

        frames = []
        for i, seed in enumerate(seeds, start=1):
            frames.append(run_one(mode, seed, i, repeats))

        raw = pd.concat(frames, ignore_index=True)
        raw.to_csv(paths["raw"], index=False)

        agg = aggregate(raw)
        agg.to_csv(paths["agg"], index=False)

        print()
        print(f"Сырые результаты: {paths['raw']}")
        print(f"Median/mean/std результаты: {paths['agg']}")

    print()
    print("Строю сводные графики (5 PNG: время, общее время, память, random, log-log)...")
    plot_files = plot_summary_charts(agg, mode, wl)

    make_summary(raw, agg, repeats, mode, wl, plot_files, paths["summary"])

    print()
    print("Готово. Файлы:")
    print(paths["raw"])
    print(paths["agg"])
    for p in plot_files:
        print(p)
    print(paths["summary"])


if __name__ == "__main__":
    main()
