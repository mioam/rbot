from openpi.policies import policy_config
from openpi.policies.my_policy import make_my_example
from openpi.shared import download
from openpi.training import config as _config
from rbot.utils.tools import timer

config = _config.get_config('pi0_my')
checkpoint_dir = download.maybe_download('checkpoints/pi0_my/my_experiment/29999')

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example.


observation = make_my_example()


action_chunk = policy.infer(observation)
for step in range(10):
    timer.tick()
    action_chunk = policy.infer(observation)['actions']
    timer.tick()
    print(action_chunk.shape)
