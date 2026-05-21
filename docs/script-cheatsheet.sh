uv run third_party/openpi/scripts/compute_norm_stats.py --config-name pi0_my
uv run third_party/openpi/scripts/compute_norm_stats.py --config-name pi0_my_lora --repo-id miaom/carrot_fix_pot



# 训练
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run third_party/openpi/scripts/train.py pi0_my --exp-name=carrot --overwrite
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run third_party/openpi/scripts/train.py pi0_my --exp-name=carrot --data.repo-id miaom/carrot_fix_pot


# 测试

uv run third_party/openpi/scripts/serve_policy.py policy:checkpoint --policy.config=pi0_my --policy.dir=checkpoints/pi0_my/my_experiment/29999

uv run third_party/openpi/scripts/serve_policy.py policy:checkpoint --policy.config=pi05_my --policy.dir=checkpoints/pi05_my/carrot_fix_pot/19999
