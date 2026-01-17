import pandas as pd
import matplotlib.pyplot as plt

def plot_soro_results(csv_path="soro100epi.csv"):#読み込むファイル名
    # CSVファイルを読み込み
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} が見つかりません。")
        return

    # グラフの設定 (2行2列のサブプロット)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Waiting Time per Intersection (Episodes 1-10)', fontsize=16)

    # 各交差点のプロット設定
    intersections = [
        {"id": "C1", "ax": axes[0, 0], "color": "blue"},
        {"id": "C2", "ax": axes[0, 1], "color": "green"},
        {"id": "C3", "ax": axes[1, 0], "color": "orange"},
        {"id": "C4", "ax": axes[1, 1], "color": "purple"}
    ]

    # データプロット
    for item in intersections:
        ax = item["ax"]
        cid = item["id"]
        
        ax.plot(df["Episode"], df[cid], marker='o', linestyle='-', color=item["color"], linewidth=2, label=cid)
        
        ax.set_title(f"Intersection {cid}", fontsize=14)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Waiting Time")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xticks(df["Episode"]) # 横軸を整数エピソードにする
        
        # 値のラベルを表示
        for x, y in zip(df["Episode"], df[cid]):
            ax.annotate(f"{int(y)}", (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # タイトル分のスペースを空ける
    
    # 保存
    output_filename = "soro_results_graph.png"
    plt.savefig(output_filename)
    print(f"グラフを保存しました: {output_filename}")
    
    # 表示 (環境によってはウィンドウが開きます)
    plt.show()

if __name__ == "__main__":
    plot_soro_results()