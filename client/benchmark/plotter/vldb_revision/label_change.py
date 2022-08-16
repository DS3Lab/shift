import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

change_exp_path = '.cache/old/changes_ours.csv'

sns.set(
    font="DejaVu Sans",
    context="paper",
    style="whitegrid",
    font_scale=2,
    rc={'figure.figsize':(9,8)}
)
sns.set_palette("colorblind")

def draw():
    baseline_df = pd.read_csv(".cache/old/e2e_100.csv")
    baseline_df = baseline_df[baseline_df['reader']=='CIFAR-100']
    baseline_df = baseline_df[baseline_df['classifier']=='Cosine NN']
    exclude_baseline_df = pd.read_csv(change_exp_path)
    baseline_8gpu_time = baseline_df[baseline_df['settings']=='w/ SH 8 GPUs']['elapsed'].mean()/3600
    baseline_1gpu_time = baseline_df[baseline_df['settings']=='w/ SH 1 GPU']['elapsed'].mean()/3600

    ax = sns.lineplot()
    ax.axhline(y=baseline_8gpu_time, color='C2', lw=4, label='Baseline 8 GPUs')
    ax.axhline(y=baseline_1gpu_time, color='C4', lw=4, label='Baseline 1 GPU')
    ax.axhline(y=0.01348/3600, color='C0', lw=4, label='w/ SHiFT', linestyle="dashed")

    box = ax.get_position()
    ax.set_xlabel("Percent of Changes")
    ax.set_ylabel("Time (Hours)")
    ax.set_yscale("log")
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])
    
    handles, labels = plt.gca().get_legend_handles_labels()
  
    # pass handle & labels lists along with order as below
    leg = ax.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, -0.1))

    leg.get_lines()[0].set_linewidth(4)
    leg.get_lines()[1].set_linewidth(4)
    leg.get_lines()[2].set_linewidth(4)
    # set different marker
    leg.get_lines()[2].set_linestyle("dashed")
    
    # plt.show()
    plt.savefig(
        'figures/shift/label_change_100.pdf', 
        bbox_inches='tight',
        pad_inches=0
    )
if __name__ == '__main__':
    draw()