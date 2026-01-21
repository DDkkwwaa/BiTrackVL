from lib.test.evaluation import get_dataset
from lib.test.analysis.vasttrack_plot_results import print_results

def main():
    results_dir = "/home/lihui/HIPB_up_large/output/test/tracking_results/hiptrack/hiptrack_train_full"   # 你的结果目录
    dataset = get_dataset("vasttrack")
    print_results(
        results_dir,
        dataset,
        merge_results=False,
        plot_types=("success", "prec", "norm_prec")
    )

if __name__ == "__main__":
    main()
