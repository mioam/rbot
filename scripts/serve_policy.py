import logging
from pathlib import Path
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from rbot.openpi.config import build_config
from rbot.openpi.config_builder import load_config


def main(ckpt: Path, default_prompt='', port=8000) -> None:
    config_raw = load_config(ckpt / '..' / 'config.yaml')
    config = build_config(config_raw)

    policy = _policy_config.create_trained_policy(
        config, ckpt, default_prompt=default_prompt
    )
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if False:
        policy = _policy.PolicyRecorder(policy, 'policy_records')

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info('Creating server (host: %s, ip: %s)', hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host='0.0.0.0',
        port=port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, force=True)

    ckpt = Path('checkpoints/pi0_jg_r_rel/exp-260528-180419/9999')

    tyro.cli(main)
