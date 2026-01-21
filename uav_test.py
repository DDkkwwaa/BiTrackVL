import os
import shutil
import subprocess
import yaml

ep = range(99, 94, -1)
path = '/data/code_Lon/PycharmProjects/HIPB_up_large'
dest_dir = os.path.join(path, f'uav')
os.makedirs(dest_dir, exist_ok=True)
for i in ep:
    source_dir = os.path.join(path, 'output/test/result_plots')
    shutil.rmtree(source_dir, ignore_errors=True)
    config_file = os.path.join(path, 'experiments/hiptrack/hiptrack.yaml')
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    config['TEST']['EPOCH'] = i
    with open(config_file, 'w') as file:
        yaml.safe_dump(config, file)
    output_dir = os.path.join(path, 'output/test/tracking_results/hiptrack/hiptrack')
    if os.path.exists(output_dir):
        subprocess.run(['rm', '-rf', output_dir], check=True)
    subprocess.run(
	'python3 tracking/test.py hiptrack hiptrack --dataset uav --threads  2 --num_gpus 1',
        shell=True,
        check=True,
        cwd=path
    )

    result = subprocess.run(
        'python3 tracking/analysis_results.py',
        shell=True,
        check=True,
        cwd=path,
        capture_output=True,
        text=True)
    output_text = result.stdout
    error_text = result.stderr
    dest_dir = os.path.join(path, f'uav/uav-ep{i}')
    os.makedirs(dest_dir, exist_ok=True)
    with open(os.path.join(dest_dir, f'output_{i}.txt'), 'w') as f_out:
        f_out.write(output_text)
    if error_text:
        with open(os.path.join(dest_dir, f'error_{i}.txt'), 'w') as f_err:
            f_err.write(error_text)

    source_dir = os.path.join(path, 'output/test/tracking_results/hiptrack/hiptrack')
    subprocess.run(['mv', source_dir, dest_dir], check=True)
    shutil.rmtree(source_dir, ignore_errors=True)