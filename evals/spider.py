import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_results(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results CSV not found at: {csv_path}")

    # Dataset key mapping to display labels
    categories = [
        ("gp_tools", "Tool Use"),
        ("gp_redteam", "Red-Teaming"),
        ("gp_hitl", "Human-in-the-Loop")
    ]

    stats = {cat_key: {"total": 0, "gemma_pass": 0, "gpt_pass": 0} for cat_key, _ in categories}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row.get("dataset", "").strip()
            gemma_res = row.get("gemma_result", "").strip().upper()
            gpt_res = row.get("gpt_result", "").strip().upper()

            if dataset in stats:
                stats[dataset]["total"] += 1
                if "PASS" in gemma_res:
                    stats[dataset]["gemma_pass"] += 1
                if "PASS" in gpt_res:
                    stats[dataset]["gpt_pass"] += 1

    labels = [label for _, label in categories]
    gemma_scores = []
    gpt_scores = []

    for cat_key, label in categories:
        total = stats[cat_key]["total"]
        if total > 0:
            g_rate = (stats[cat_key]["gemma_pass"] / total) * 100.0
            gpt_rate = (stats[cat_key]["gpt_pass"] / total) * 100.0
        else:
            g_rate = 0.0
            gpt_rate = 0.0
        gemma_scores.append(g_rate)
        gpt_scores.append(gpt_rate)

    return labels, gemma_scores, gpt_scores, stats

def create_spider_plot(labels, gemma_scores, gpt_scores, output_path="evals/spider_plot.png"):
    num_vars = len(labels)
    # Compute angles for each vertex (3 vertices = equilateral triangle)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Complete the loop for radar chart
    gemma_plot = gemma_scores + gemma_scores[:1]
    gpt_plot = gpt_scores + gpt_scores[:1]
    angles_plot = angles + angles[:1]

    # Setup styling
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Orient top vertex to the top center
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Set category labels
    plt.xticks(angles, labels, color='#1f2328', size=13, weight='bold')

    # Radial / Y-axis setup
    ax.set_rlabel_position(30)
    plt.yticks([25, 50, 75, 100], ["25%", "50%", "75%", "100%"], color="#656d76", size=10)
    plt.ylim(0, 105)

    # Plot Gemma 4 (31B)
    ax.plot(
        angles_plot,
        gemma_plot,
        color='#1f77b4',
        linewidth=2.5,
        linestyle='solid',
        marker='o',
        markersize=7,
        label=f'Gemma 4: 31B (Avg: {np.mean(gemma_scores):.1f}%)'
    )
    ax.fill(angles_plot, gemma_plot, color='#1f77b4', alpha=0.22)

    # Plot GPT OSS (120B)
    ax.plot(
        angles_plot,
        gpt_plot,
        color='#2ca02c',
        linewidth=2.5,
        linestyle='--',
        marker='s',
        markersize=7,
        label=f'GPT-OSS: 120B (Avg: {np.mean(gpt_scores):.1f}%)'
    )
    ax.fill(angles_plot, gpt_plot, color='#2ca02c', alpha=0.18)

    # Add title and legend
    plt.title("Harness Evaluation Benchmark\n(Tri-Axis Capabilities)", size=16, weight='bold', y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), frameon=True, fontsize=11)
    plt.tight_layout()

    # Ensure parent directory exists
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Spider plot successfully saved to: {output_path}")

def main():
    default_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv")
    default_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spider_plot.png")

    parser = argparse.ArgumentParser(description="Generate a 3-axis spider plot from evaluation results.")
    parser.add_argument("--results", type=str, default=default_csv, help="Path to results.csv")
    parser.add_argument("--output", type=str, default=default_output, help="Path to save output plot (e.g. spider_plot.png)")
    args = parser.parse_args()

    labels, gemma_scores, gpt_scores, stats = load_results(args.results)

    print("\n=== Evaluation Results Summary ===")
    for (dataset, (cat_key, label)), g_score, gpt_score in zip(
        [("gp_tools", ("gp_tools", "Tool Use")), ("gp_redteam", ("gp_redteam", "Red-Teaming")), ("gp_hitl", ("gp_hitl", "Human-in-the-Loop"))],
        gemma_scores,
        gpt_scores
    ):
        total = stats[cat_key]["total"]
        g_pass = stats[cat_key]["gemma_pass"]
        gpt_pass = stats[cat_key]["gpt_pass"]
        print(f"• {label:20s}: Gemma = {g_pass}/{total} ({g_score:.1f}%), GPT = {gpt_pass}/{total} ({gpt_score:.1f}%)")

    create_spider_plot(labels, gemma_scores, gpt_scores, args.output)

if __name__ == "__main__":
    main()
