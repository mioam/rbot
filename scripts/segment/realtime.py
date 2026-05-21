from concurrent import futures
import logging
import os

import grpc
from myrpc import rpc_pb2, rpc_pb2_grpc
from myrpc.rpc_utils import from_bytes, to_bytes
import numpy as np
from sam3.model_builder import build_sam3_stream_predictor
import torch

os.environ['HF_ENDPOINT'] = 'http://hf-mirror.com'
# from sam2.build_sam import build_sam2_video_predictor, build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
# from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
# from utils.track_utils import sample_points_from_masks
# from utils.video_utils import create_video_from_images


# sam3 predictor builder (used in scripts/inference/video_stream.py)

LOG = logging.getLogger('sam3_server')
LOG.setLevel(logging.INFO)


class Sam3Core:
    """Wrapper around the sam3 stream predictor to provide simple session-based APIs."""

    def __init__(self, device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        LOG.info(f'Building sam3 predictor on device: {self.device}')
        self.predictor = build_sam3_stream_predictor(
            device=self.device, checkpoint_path='/ssd1/mzc/workspace/sam3/ckpts/sam3.pt'
        )
        # Keep track of session ids that were created via predictor
        # self.sessions = set()

    def start_session(self):
        resp = self.predictor.handle_request({'type': 'start_session'})
        session_id = resp.get('session_id')
        if session_id is None:
            raise RuntimeError('Failed to start session (no session_id returned).')
        # self.sessions.add(session_id)
        return session_id

    def add_frame(self, session_id, frame_rgb: np.ndarray):
        # predictor expects numpy RGB arrays, same as video_stream script
        return self.predictor.handle_request({
            'type': 'add_frame',
            'session_id': session_id,
            'frame': frame_rgb,
        })

    def add_prompt(self, session_id, frame_index: int, text: str):
        return self.predictor.handle_request({
            'type': 'add_prompt',
            'session_id': session_id,
            'frame_index': frame_index,
            'text': text,
        })

    def run_inference(self, session_id, frame_index: int):
        ret = self.predictor.handle_request({
            'type': 'run_inference',
            'session_id': session_id,
            'frame_index': frame_index,
        })
        # self.predictor._get_session(session_id)['state']['text_prompt'] = None
        return ret

    def end_session(self, session_id):
        # best-effort cleanup if predictor implements it; ignore if not supported
        try:
            resp = self.predictor.handle_request({
                'type': 'close_session',
                'session_id': session_id,
            })
        except Exception:
            resp = None
        # self.sessions.discard(session_id)
        return resp


class SAM3Server(rpc_pb2_grpc.RPCServicer):
    """gRPC wrapper that supports 'one_image', 'new_video', and 'video' request types."""

    def __init__(self, core: Sam3Core):
        self.core = core
        self.session_id = None

    def _parse_outputs(self, outputs):
        print(outputs)
        # outputs format is the same as run_single_frame_inference postprocess in sam3 code:
        # expected keys: out_binary_masks (N,H,W), out_obj_ids, out_probs, out_boxes_xywh
        if outputs is None:
            return np.zeros((0,)), [], {}
        # numpy array boolean (N,H,W) ideally
        masks = outputs.get('out_binary_masks')
        obj_ids = outputs.get('out_obj_ids')  # numpy array
        boxes = outputs.get('out_boxes_xywh')
        probs = outputs.get('out_probs')
        labels = obj_ids.tolist()
        meta = {'boxes': boxes, 'probs': probs}
        return masks, labels, meta

    @torch.inference_mode()
    @torch.autocast('cuda', dtype=torch.bfloat16)
    def Get(self, request, context):
        data = from_bytes(request.data)
        rtype = data.get('type', 'one_image')
        try:
            if rtype == 'one_image':
                # transient session: start -> add_frame -> add_prompt(optional) -> run -> (optionally end)
                rgb = data['rgb_image_numpy']
                prompt = data.get('prompt_text', None)
                session_id = self.core.start_session()
                # add frame
                self.core.add_frame(session_id, rgb)
                if prompt:
                    resp = self.core.add_prompt(session_id, frame_index=0, text=prompt)
                # self.core.run_inference(session_id, frame_index=0)
                outputs = resp.get('outputs')
                masks, labels, meta = self._parse_outputs(outputs)

                # best-effort cleanup
                self.core.end_session(session_id)
                return rpc_pb2.Reply(
                    data=to_bytes({'masks': masks, 'labels': labels, 'meta': meta})
                )

            if rtype == 'new_video':
                if self.session_id is not None:
                    self.core.end_session(self.session_id)
                # start a persistent session for streaming: returns session_id
                rgb = data['rgb_image_numpy']
                prompts = data.get('prompt_text', None)
                self.session_id = self.core.start_session()
                # add first frame
                self.core.add_frame(self.session_id, rgb)
                if prompts:
                    prompt_list = [x.strip() for x in prompts.split('.')]
                    for prompt in prompt_list:
                        if prompt != '':
                            self.core.add_prompt(
                                self.session_id, frame_index=0, text=prompt
                            )
                resp = self.core.run_inference(self.session_id, frame_index=0)
                outputs = resp.get('outputs')
                masks, labels, meta = self._parse_outputs(outputs)
                return rpc_pb2.Reply(
                    data=to_bytes({'masks': masks, 'labels': labels, 'meta': meta})
                )

            if rtype == 'video':
                # append a frame to an existing session and run inference
                session_id = self.session_id
                rgb = data['rgb_image_numpy']

                frame_index = self.core.add_frame(session_id, rgb)['frame_index']
                # If client provided a text prompt to add at this frame:
                # prompt = data.get("prompt_text", None)
                # if prompt is not None:
                #     self.core.add_prompt(
                #         session_id, frame_index=frame_index, text=prompt)

                resp = self.core.run_inference(session_id, frame_index=frame_index)
                outputs = resp.get('outputs')
                masks, labels, meta = self._parse_outputs(outputs)
                return rpc_pb2.Reply(
                    data=to_bytes({'masks': masks, 'labels': labels, 'meta': meta})
                )

            return rpc_pb2.Reply(
                data=to_bytes({'error': f'Unknown request type: {rtype}'})
            )
        except Exception as e:
            LOG.exception('Exception handling request')
            return rpc_pb2.Reply(data=to_bytes({'error': str(e)}))


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    core = Sam3Core(device=device)
    port = os.environ.get('SAM3_SERVER_PORT', '50050')
    MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ],
    )
    rpc_pb2_grpc.add_RPCServicer_to_server(SAM3Server(core), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    LOG.info(f'SAM3 gRPC server started on port {port}')
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        LOG.info('Shutting down server...')


if __name__ == '__main__':
    main()
