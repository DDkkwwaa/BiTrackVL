#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
目标跟踪模型调参测试脚本
用于批量测试不同epoch权重在otb数据集上的性能

Ablation Experiment 2.1
"""

import os
import re
import subprocess
import shutil
import sys
from pathlib import Path


class TrackingTester:
    def __init__(self, project_dir="/data/code_Lon/PycharmProjects/HIPB_up_large"):
        self.project_dir = Path(project_dir)
        self.param_file = self.project_dir / "lib/test/parameter/hiptrack.py"
        self.test_script = self.project_dir / "tracking/test.py"
        self.analysis_script = self.project_dir / "tracking/analysis_results.py"
        self.results_dir = self.project_dir / "output/test/tracking_results/hiptrack/hiptrack"
        self.result_plots_dir = self.project_dir / "output/test/result_plots"

        # 检查必要文件是否存在
        self._check_files()

    def _check_files(self):
        """检查必要的文件和目录是否存在"""
        required_files = [self.param_file, self.test_script, self.analysis_script]
        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f"找不到必要文件: {file_path}")

        # 确保结果目录存在
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def modify_epoch_param(self, new_epoch: int) -> bool:
        with open(self.param_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if 'params.checkpoint' in line:
                lines[i] = f'    params.checkpoint = os.path.join(save_dir, "checkpoints/train/hiptrack/hiptrack/HIPTrack_ep{new_epoch:04d}.pth.tar")\n'
                with open(self.param_file, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                print(f"已强制改为 epoch {new_epoch:04d}")
                return True
        return False

    def run_test(self, dataset_name="otb"):
        """运行测试脚本"""
        print(f"开始在{dataset_name}数据集上运行测试...")

        cmd = [
            sys.executable, str(self.test_script),
             "hiptrack",
             "hiptrack",
            "--dataset_name", dataset_name,
            "--threads", "1"
        ]

        try:
            # 切换到项目目录
            result = subprocess.run(
                cmd,
                cwd=self.project_dir,
                check=True,
                capture_output=False,
                text=True
            )
            print("测试运行完成")
            return True
        except subprocess.CalledProcessError as e:
            print(f"测试运行失败: {e}")
            print(f"错误输出: {e.stderr}")
            return False

    def run_analysis(self):
        """运行结果分析脚本并保存输出"""
        print("开始运行结果分析...")

        # 清理之前的结果缓存
        if self.result_plots_dir.exists():
            shutil.rmtree(self.result_plots_dir)
            print("已清理result_plots缓存")

        cmd = [sys.executable, str(self.analysis_script)]

            # 切换到项目目录并运行分析脚本
        result = subprocess.run(
            cmd,
            cwd=self.project_dir,
            check=True,
            capture_output=True,
            text=True
        )

        # 保存命令行输出到_result.txt
        result_file = self.results_dir / "_result.txt"
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write("=== 标准输出 ===\n")
            f.write(result.stdout)
            f.write("\n=== 错误输出 ===\n")
            f.write(result.stderr)

        print(f"分析完成，结果已保存到: {result_file}")
        return True

    def check_result_exists(self, dataset_name, epoch_num):
        """检查指定epoch的测试结果是否已存在"""
        result_folder_name = f"{dataset_name}_HIPTrack_ep{epoch_num:04d}"
        result_folder = self.results_dir / result_folder_name
        return result_folder.exists()

    def save_results(self, dataset_name, epoch_num):
        """保存测试结果到指定文件夹"""
        result_folder_name = f"{dataset_name}_HIPTrack_ep{epoch_num:04d}"
        result_folder = self.results_dir / result_folder_name

        # 创建结果文件夹
        result_folder.mkdir(exist_ok=True)
        print(f"创建结果文件夹: {result_folder}")

        # 移动所有txt文件到结果文件夹
        moved_files = []
        for txt_file in self.results_dir.glob("*.txt"):
            if txt_file.parent != result_folder:  # 避免移动自己
                dest_file = result_folder / txt_file.name
                shutil.move(str(txt_file), str(dest_file))
                moved_files.append(txt_file.name)

        print(f"已移动 {len(moved_files)} 个文件到结果文件夹")
        if moved_files:
            print(f"移动的文件: {', '.join(moved_files[:5])}{'...' if len(moved_files) > 5 else ''}")

        return result_folder

    def test_single_epoch(self, epoch_num, dataset_name="otb"):
        """测试单个epoch的完整流程"""
        print(f"\n{'=' * 50}")
        print(f"开始测试 Epoch {epoch_num:04d} on {dataset_name}")
        print(f"{'=' * 50}")

        # 步骤0: 检查结果是否已存在
        if self.check_result_exists(dataset_name, epoch_num):
            result_folder_name = f"{dataset_name}_HIPTrack_ep{epoch_num:04d}"
            print(f"检测到结果文件夹 {result_folder_name} 已存在，跳过此epoch")
            return "skipped"

        # 步骤1: 修改epoch参数
        if not self.modify_epoch_param(epoch_num):
            print("修改epoch参数失败，跳过此epoch")
            return False

        # 步骤2: 运行测试
        if not self.run_test(dataset_name):
            print("测试运行失败，跳过此epoch")
            return False

        # 步骤3: 运行分析
        if not self.run_analysis():
            print("结果分析失败，跳过此epoch")
            return False

        # 步骤4: 保存结果
        result_folder = self.save_results(dataset_name, epoch_num)

        print(f"Epoch {epoch_num:04d} 测试完成，结果保存在: {result_folder}")
        return True

    def test_epoch_range(self, start_epoch, end_epoch, dataset_name="otb"):
        """批量测试epoch范围"""
        print(f"开始批量测试 Epoch {start_epoch:04d} 到 {end_epoch:04d}")

        successful_tests = []
        failed_tests = []
        skipped_tests = []

        for epoch in range(start_epoch, end_epoch + 1):
            try:
                result = self.test_single_epoch(epoch, dataset_name)
                if result == "skipped":
                    skipped_tests.append(epoch)
                elif result:
                    successful_tests.append(epoch)
                else:
                    failed_tests.append(epoch)
            except KeyboardInterrupt:
                print("\n用户中断测试")
                break
            except Exception as e:
                print(f"测试 Epoch {epoch:04d} 时发生异常: {e}")
                failed_tests.append(epoch)

        # 输出总结
        print(f"\n{'=' * 50}")
        print("批量测试完成")
        print(f"{'=' * 50}")
        print(f"成功测试的epochs: {successful_tests}")
        print(f"跳过测试的epochs: {skipped_tests}")
        print(f"失败测试的epochs: {failed_tests}")

        total_epochs = len(successful_tests) + len(failed_tests) + len(skipped_tests)
        print(f"总计: {total_epochs} 个epochs")
        print(f"  - 成功: {len(successful_tests)} 个")
        print(f"  - 跳过: {len(skipped_tests)} 个")
        print(f"  - 失败: {len(failed_tests)} 个")

        if len(successful_tests) + len(skipped_tests) > 0:
            completion_rate = (len(successful_tests) + len(skipped_tests)) / total_epochs * 100
            print(f"完成率: {completion_rate:.1f}%")

        if failed_tests:
            print(f"\n需要重新测试的epochs: {failed_tests}")

        return {
            'successful': successful_tests,
            'skipped': skipped_tests,
            'failed': failed_tests
        }


def main():
    """主函数"""
    try:
        # 创建测试器实例
        tester = TrackingTester()

        # 测试epoch30到epoch50
        tester.test_epoch_range(80, 90, "otb")

    except Exception as e:
        print(f"程序运行出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()