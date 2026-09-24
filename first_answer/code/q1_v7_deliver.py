"""Data-derived Q1 tables and six-page vector PDF; no final-paper claims."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from q1_v7_core import config, folder, save_json, save_table
from utils import PROJECT_ROOT, choose_chinese_font, sha256_file

BLUE, ORANGE, GREEN = "#2374AB", "#D56B26", "#19856C"
METHODS = ["P0", "W0", "SELECTED"]
COLORS = dict(zip(METHODS, [BLUE, GREEN, ORANGE]))
FAMILIES = {"broad_common": "宽共同峰", "narrow_antisym": "窄反对称峰",
            "shifted_antisym": "延迟峰", "biphasic_common": "双相波",
            "oscillatory": "振荡波", "slow_common": "慢共同波",
            "outside_subspace_stress": "子空间外", "mixed_subspace_stress": "混合方向"}


def read(name):
    return pd.read_csv(folder()/name, float_precision="round_trip")


def summary():
    prediction = read("heldout_erp_prediction.csv")
    prediction["scope"] = np.where(prediction.partition == "historical", "historical", "development_outer")
    prediction = prediction.groupby(["scope", "task", "method"], as_index=False).agg(
        mean_scaled_rmse=("scaled_rmse", "mean"), mean_correlation=("correlation", "mean"))
    save_table("prediction_summary.csv", prediction)
    stability = read("split_half_metrics.csv").groupby(["task", "method", "contrast"], as_index=False).agg(
        mean_correlation=("correlation", "mean"), median_correlation=("correlation", "median"),
        mean_rmse=("rmse", "mean"))
    save_table("stability_summary.csv", stability)
    params = read("gaussian_parameters.csv")
    fits = params[["task", "condition", "channel", "method", "selected_k", "converged", "boundary",
                   "train_rmse", "historical_P0_rmse", "successful_bootstraps"]].drop_duplicates()
    save_table("gaussian_fit_summary.csv", fits)
    inj = read("injection_metrics.csv")
    main_inj = inj[(inj.scope == "single_trial_operator") & inj.family.isin(["broad_common", "narrow_antisym"])
                   & ((inj.method == "P0") | inj.is_selected)]
    save_table("primary_injection_summary.csv", main_inj)
    stats = read("response_statistics.csv")
    primary = stats[stats.primary_family]
    save_table("primary_response_summary.csv", primary)
    save_json("numerical_summary.json", {
        "scope": "E0 and Q1 only; Q2/Q3 v7 not run", "trial_count": 400,
        "prediction": prediction.to_dict("records"), "stability": stability.to_dict("records"),
        "primary_injection": main_inj.to_dict("records"),
        "primary_response": primary.to_dict("records"),
        "fit_count": len(fits), "fit_complexities": fits.selected_k.value_counts().sort_index().to_dict(),
        "fit_boundary_count": int(fits.boundary.sum()), "all_fit_converged": bool(fits.converged.all()),
        "gate": json.loads((folder()/"q1_interface_gate.json").read_text(encoding="utf-8")),
        "interpretation": "No robust P300 identification: common effect is filter-dependent; no supported left/right difference. P0 retained as reference, not declared lossless."})


def setup():
    font = choose_chinese_font()
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9,
                         "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.unicode_minus": False,
                         "savefig.facecolor": "white"})
    return font


def finish(pdf, fig, page, note):
    fig.subplots_adjust(left=.085, right=.97, bottom=.16, top=.93, hspace=.52, wspace=.31)
    fig.text(.085, .075, note, fontsize=8, va="top", linespacing=1.6)
    fig.text(.97, .035, f"Q1 · v7.1  |  {page}/6", ha="right", color="#666666", fontsize=8)
    pdf.savefig(fig)
    plt.close(fig)


def wave_axes(ax, title):
    ax.set_title(title, loc="left")
    ax.axhline(0, color="#B5B5B5", lw=.7)
    ax.axvline(0, color="#B5B5B5", lw=.7)
    ax.set_xlabel("提示起点后的时间 / s")
    ax.set_ylabel("幅值 / 记录单位")
    ax.set_xlim(-.2, .8)


def figures():
    font = setup()
    target = PROJECT_ROOT/config()["figure_dir"]
    target.mkdir(parents=True, exist_ok=True)
    path = target/"q1_results.pdf"
    erp, fits = read("erp_curves.csv"), read("gaussian_fitted_curves.csv")
    fits_meta = read("gaussian_fit_summary.csv")
    injection, curves = read("injection_metrics.csv"), read("injection_curves.csv")
    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(11.7, 8.3))
        for row, task in enumerate([1, 2]):
            for col, channel in enumerate(config()["channels"]):
                ax = axes[row, col]
                for sign, label, color in [(-1, "左条件", BLUE), (1, "右条件", ORANGE)]:
                    g = erp[(erp.task == task)&(erp.channel == channel)&(erp.condition == sign)&(erp.method == "P0")]
                    ax.plot(g.time_s, g["mean"], color=color, label=label, lw=1.3)
                    ax.fill_between(g.time_s.to_numpy(), g.ci_low.to_numpy(), g.ci_high.to_numpy(), color=color, alpha=.14)
                ax.axvspan(.25, .5, color="#777777", alpha=.07)
                wave_axes(ax, f"任务 {task} · {channel}")
                ax.legend(frameon=False)
        finish(pdf, fig, 1, "数据：400次提示锁定试验；P0 = 0.5–30 Hz 带通 + 刺激前基线。阴影为500次记录内块重采样的逐点95%区间。\n灰色时间窗为预先固定的250–500 ms；左右条件分别绘制，但不据此直接命名P300。物理幅值换算未确认。")

        for task in [1, 2]:
            fig, axes = plt.subplots(2, 3, figsize=(11.7, 8.3))
            for row, sign in enumerate([-1, 1]):
                for col, channel in enumerate(config()["channels"]):
                    ax = axes[row, col]
                    g = fits[(fits.task == task)&(fits.condition == sign)&(fits.channel == channel)]
                    m = fits_meta[(fits_meta.task == task)&(fits_meta.condition == sign)&(fits_meta.channel == channel)].iloc[0]
                    ax.plot(g.time_s, g.training_clean_mean, color=BLUE, label="开发集均值", lw=1.1)
                    ax.plot(g.time_s, g.historical_P0_mean, color="#989898", label="历史测试均值", lw=.9, ls="--")
                    ax.plot(g.time_s, g["fit"], color=ORANGE, label="曲线模型", lw=1.5)
                    ax.fill_between(g.time_s.to_numpy(), g.fit_ci_low.to_numpy(), g.fit_ci_high.to_numpy(), color=ORANGE, alpha=.15)
                    sign_name = "左" if sign == -1 else "右"
                    boundary = "；触边" if m.boundary else ""
                    wave_axes(ax, f"任务 {task} · {sign_name} · {channel}  (K={m.selected_k}{boundary})")
                    ax.legend(frameon=False, ncol=1, fontsize=7)
            finish(pdf, fig, task+1, "K=0/1/2 分别为常数/单高斯/双高斯；仅在开发集四折中选择，最终接口为P0。阴影为固定K的100次块重采样逐点区间。\n历史测试段已在旧版本中查看，不是新盲测。参数触边或幅值区间跨零时，不对峰时、峰幅作精确生理解释。")

        fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))
        for col, task in enumerate([1, 2]):
            ax = axes[0, col]
            subset = injection[(injection.task == task)&(injection.scope == "single_trial_operator")]
            selected = subset[subset.is_selected].method.iloc[0]
            for offset, name, color in [(-.17, "P0", BLUE), (.17, selected, ORANGE)]:
                g = subset[subset.method == name].set_index("family").loc[list(FAMILIES)]
                ax.bar(np.arange(8)+offset, 100*g.amplitude_bias_abs.to_numpy(), .31, color=color, label=name)
            ax.axhline(10, color="#555555", ls="--", lw=.8, label="预设10%保真线")
            ax.set_xticks(np.arange(8), list(FAMILIES.values()), rotation=28, ha="right")
            ax.set_ylabel("最大通道峰幅偏差 / %")
            ax.set_title(f"任务 {task} · 已知波形恢复", loc="left")
            ax.legend(frameon=False, ncol=3, fontsize=7)
            ax = axes[1, col]
            trade = injection[(injection.task == task)&(injection.scope == "weighted_ERP_pipeline")
                              &(injection.family == "broad_common")]
            for artifact, marker, label in [("blink_like", "o", "眨眼样"), ("muscle_like", "s", "肌电样"), ("step_like", "^", "阶跃样")]:
                g = trade[trade.artifact == artifact]
                ax.scatter(g.relative_waveform_error*100, g.artifact_removal_fraction*100,
                           color="#B8BDC3", marker=marker, s=22)
                for name, color in [("P0", BLUE), (selected, ORANGE)]:
                    r = g[g.method == name].iloc[0]
                    ax.scatter(r.relative_waveform_error*100, r.artifact_removal_fraction*100, color=color, marker=marker, s=55,
                               label=f"{name} · {label}")
            ax.set_xlabel("注入信号相对波形误差 / %（越小越好）")
            ax.set_ylabel("人工伪迹残差能量降低 / %")
            ax.set_title(f"任务 {task} · 信号保留与人工伪迹权衡", loc="left")
            ax.legend(frameon=False, fontsize=6.5, ncol=2)
        finish(pdf, fig, 4, "上图以滤波前注入真值为基准，不用平滑输出自证保真；子空间外/混合方向仅为压力测试。两项候选均未通过窄峰10%门槛。\n下图为人工伪迹且质量权重会重算；灰点为其余固定候选。去除率可为负，不能解释为真实记录的伪迹去除率。")

        fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))
        primary = read("primary_response_summary.csv")
        for col, contrast in enumerate(["common", "difference"]):
            ax = axes[0, col]
            g = primary[primary.contrast == contrast].sort_values(["task", "channel"]).reset_index(drop=True)
            for i, r in g.iterrows():
                color = BLUE if r.task == 1 else ORANGE
                ax.hlines(i, r.ci_low, r.ci_high, color=color, lw=2)
                ax.scatter(r.effect, i, color=color, marker="o" if r.p_holm_conditional < .05 else "x", s=35)
            ax.axvline(0, color="#888888", lw=.8)
            ax.set_yticks(np.arange(6), [f"任务{r.task} · {r.channel}" for r in g.itertuples()])
            ax.invert_yaxis()
            ax.set_xlabel("固定晚期窗效应 / 记录单位")
            ax.set_title("共同响应 (左+右)/2" if contrast == "common" else "条件差异 左−右", loc="left")
        stability = read("stability_summary.csv")
        prediction = read("prediction_summary.csv")
        for ax, table, value, title in [(axes[1, 0], stability, "mean_correlation", "独立分半稳定性"),
                                         (axes[1, 1], prediction[prediction.scope == "development_outer"], "mean_scaled_rmse", "开发集外折ERP预测")]:
            groups = [(1, "common"), (1, "difference"), (2, "common"), (2, "difference")] if value == "mean_correlation" else [(1, None), (2, None)]
            for j, method in enumerate(METHODS):
                ys = []
                for task, contrast in groups:
                    part = table[(table.task == task)&(table.method == method)]
                    if contrast is not None:
                        part = part[part.contrast == contrast]
                    ys.append(part[value].iloc[0])
                ax.bar(np.arange(len(groups))+(j-1)*.22, ys, .20, color=COLORS[method], label=method)
            labels = [f"任务{task}\n"+("共同" if c == "common" else "差分" if c else "") for task, c in groups]
            ax.set_xticks(np.arange(len(groups)), labels)
            ax.axhline(0, color="#888888", lw=.8)
            ax.set_ylabel("平均相关系数" if value == "mean_correlation" else "平均通道标准化RMSE")
            ax.set_title(title, loc="left")
            ax.legend(frameon=False, ncol=3, fontsize=7)
        finish(pdf, fig, 5, "上图：10试次块，500次块重采样区间；实心圆仅表示固定P0下12项Holm校正后的条件性显著，×为其余项。区间与检验不等价。\n下图：20次独立拟合分半；共同/差分分列。外折误差越低越好；平滑候选没有稳定优于P0，共同稳定不等于左右可区分。")

        fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))
        branch = read("branch_mean_curves.csv")
        for name, color, label in [("P0", BLUE, "P0"), ("SLOW30", GREEN, "仅30 Hz低通"), ("P0_LARGER_PADDING", ORANGE, "P0扩大填充")]:
            g = branch[(branch.branch == name)&(branch.task == 2)&(branch.channel == "Fz")]
            axes[0, 0].plot(g.time_s, g.common, color=color, label=label)
        wave_axes(axes[0, 0], "任务2 · Fz共同曲线对预处理敏感")
        axes[0, 0].legend(frameon=False)
        sensitivity = read("branch_response_sensitivity.csv")
        g = sensitivity[(sensitivity.task == 2)&(sensitivity.channel == "Fz")&(sensitivity.contrast == "common")]
        for i, r in enumerate(g.itertuples()):
            color = [BLUE, GREEN, ORANGE][i]
            axes[0, 1].hlines(i, r.ci_low, r.ci_high, color=color, lw=2)
            axes[0, 1].scatter(r.effect, i, color=color)
            axes[0, 1].annotate(f"p={r.p_holm_within_branch:.3f}", (r.effect, i), xytext=(0, 8), textcoords="offset points", fontsize=8)
        axes[0, 1].set_yticks(range(3), ["P0", "仅低通", "扩大填充"])
        axes[0, 1].set_ylim(-.6, 2.7)
        axes[0, 1].axvline(0, color="#888888", lw=.8)
        axes[0, 1].set_xlabel("晚期共同效应 / 记录单位")
        axes[0, 1].set_title("效应方向及显著性不稳健", loc="left")
        slow = read("filter_fidelity_sensitivity.csv")
        for j, name in enumerate(["P0", "SLOW30"]):
            g = slow[slow.branch == name]
            axes[1, 0].bar(np.arange(len(g))+(j-.5)*.3, g.amplitude_bias_abs*100, .28,
                           color=[BLUE, GREEN][j], label=name)
        axes[1, 0].set_xticks(np.arange(6), [FAMILIES[f] for f in g.family], rotation=25, ha="right")
        axes[1, 0].set_ylabel("最大通道峰幅偏差 / %")
        axes[1, 0].set_title("保留慢波与去除漂移并非同一件事", loc="left")
        axes[1, 0].legend(frameon=False)
        padding = read("padding_sensitivity.csv")
        axes[1, 1].boxplot([padding[padding.task == task].training_scaled_rmse for task in [1, 2]],
                           labels=["任务1", "任务2"], showfliers=True,
                           flierprops={"markersize": 2, "markeredgecolor": "#999999"})
        axes[1, 1].set_ylabel("逐试次/通道标准化RMSE")
        axes[1, 1].set_title("扩大滤波填充造成的曲线变化", loc="left")
        finish(pdf, fig, 6, "敏感性分支仅用于揭示依赖，不按p值选最佳滤波。仅低通分支未去慢漂移，其正响应也不能直接认证P300。\nP0仍为后续可比参考，不是无损真值；下一阶段须保留固定预处理对照。p值仅在每个分支内作12项Holm校正。")
    save_json("figure_manifest.json", {"path": str(path.relative_to(PROJECT_ROOT)), "pages": 6,
                                      "font": font, "sha256": sha256_file(path),
                                      "sources": [str(p.relative_to(PROJECT_ROOT)) for p in sorted(folder().glob("*.csv"))]})
    print(path, flush=True)


if __name__ == "__main__":
    summary()
    figures()
