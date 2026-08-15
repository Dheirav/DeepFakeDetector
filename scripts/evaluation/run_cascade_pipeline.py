import os
import sys
import argparse
import subprocess


def run(cmd, env=None):
    print("\n> "+" ".join(cmd))
    subprocess.check_call(cmd, env=env)


def main():
    parser = argparse.ArgumentParser(description="Run cascade pipeline sequentially (Stage1 -> select -> Stage2 -> merge)")
    parser.add_argument('--stage1_model', required=True)
    parser.add_argument('--stage2_model', required=True)
    parser.add_argument('--data_dir', default='dataset_builder/test')
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--save_root', default='results/cascade_run')
    parser.add_argument('--attention_head', default='none', choices=['none','gem','cbam'])
    parser.add_argument('--python', default=sys.executable, help='Python executable to run scripts')

    args = parser.parse_args()

    stage1_dir = os.path.join(args.save_root, 'cascade_stage1')
    stage2_dir = os.path.join(args.save_root, 'cascade_stage2')
    final_dir = os.path.join(args.save_root, 'cascade_final')

    os.makedirs(stage1_dir, exist_ok=True)
    os.makedirs(stage2_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    env = os.environ.copy()

    # Stage-1: run in a separate process (can force CPU by setting CUDA_VISIBLE_DEVICES="")
    cmd1 = [args.python, 'scripts/evaluation/evaluate_stage1.py',
            '--model_path', args.stage1_model,
            '--data_dir', args.data_dir,
            '--save_dir', stage1_dir,
            '--batch_size', str(args.batch_size),
            '--num_workers', str(args.num_workers),
            '--attention_head', args.attention_head]

    run(cmd1, env=env)

    # Select subset for stage2
    cmd2 = [args.python, 'scripts/evaluation/select_stage2_subset.py',
            '--stage1_probs', os.path.join(stage1_dir, 'stage1_probs.npy'),
            '--threshold', str(args.threshold),
            '--save_dir', stage2_dir]
    run(cmd2, env=env)

    # Stage-2: run only on selected indices
    cmd3 = [args.python, 'scripts/evaluation/evaluate_stage2.py',
            '--stage2_model_path', args.stage2_model,
            '--data_dir', args.data_dir,
            '--indices_file', os.path.join(stage2_dir, 'stage2_indices.npy'),
            '--save_dir', stage2_dir,
            '--batch_size', str(args.batch_size),
            '--num_workers', str(args.num_workers),
            '--attention_head', args.attention_head]
    run(cmd3, env=env)

    # Merge
    cmd4 = [args.python, 'scripts/evaluation/merge_cascade_results.py',
            '--stage1_dir', stage1_dir,
            '--stage2_dir', stage2_dir,
            '--save_dir', final_dir]
    run(cmd4, env=env)

    print(f"\nPipeline completed. Final results in: {final_dir}")


if __name__ == '__main__':
    main()
