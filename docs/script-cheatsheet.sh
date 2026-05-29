uv run scripts/compute_norm_stats.py PI0 --data.repo_id miaom/carrot_fix_pot --data.state jg --data.action r --data.delta True --exp_name exp-260528-180419

CUDA_VISIBLE_DEVICES=1,2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py PI0 --data.repo_id miaom/carrot_fix_pot --data.state jg --data.action r --data.delta True --exp_name exp-260528-180419

# 测试

uv run third_party/openpi/scripts/serve_policy.py policy:checkpoint --policy.config=pi0_my --policy.dir=checkpoints/pi0_my/my_experiment/29999

uv run third_party/openpi/scripts/serve_policy.py policy:checkpoint --policy.config=pi05_my --policy.dir=checkpoints/pi05_my/carrot_fix_pot/19999

