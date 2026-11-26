import multiprocessing as mp
import signal
import time

from input_process.sequence import Frame

VIEWER_PUSH_INTERVAL = 0.5  # seconds between full map updates sent to the viewer


class GaussianMapper:
    """Builds the Gaussian map in a separate process.

    Follows the same process+queue pattern as SlamRenderer. The pipeline submits
    posed keyframes (with depth); the mapper process inserts them into the map and
    publishes the result to the web viewer. On stop the map is written to
    `output` as a standard 3DGS .ply.
    """

    def __init__(self, stride: int = 4, output: str = None):
        self.queue = mp.Queue(maxsize=8)
        self.process = mp.Process(target=_mapper_process_entry, args=(self.queue, stride, output))

    def start(self):
        self.process.start()

    def submit(self, keyframe: Frame):
        """Queue a keyframe. Blocks when the mapper is behind, so no keyframe is dropped."""
        self.queue.put(keyframe)

    def stop(self):
        """Finish queued keyframes, save the map and shut the viewer down."""
        self.queue.put(None)
        self.process.join()


# ---------------------------------------------------------------------------
# Everything below runs in the child process.
# ---------------------------------------------------------------------------

def _mapper_process_entry(queue, stride, output):
    # Ctrl+C reaches the whole process group; the parent decides when we stop,
    # so the map still gets saved.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    from mapping.gaussian_map import GaussianMap
    from mapping.gs_viewer import GsViewer

    gaussian_map = GaussianMap(stride=stride)
    viewer = GsViewer()
    print("[mapper] viewer: http://localhost:8080")

    num_keyframes = 0
    last_push = 0.0
    while True:
        keyframe = queue.get()
        if keyframe is None:
            break
        added, removed = gaussian_map.insert(keyframe)
        num_keyframes += 1
        viewer.set_camera(keyframe.T_wc, keyframe.image_bgr, keyframe.K)
        # the whole map is re-sent, so limit how often; always send once caught up
        if queue.empty() or time.time() - last_push > VIEWER_PUSH_INTERVAL:
            viewer.set_map(gaussian_map.means, gaussian_map.rgbs, gaussian_map.scales)
            last_push = time.time()
        print(f"[mapper] keyframe {num_keyframes}: +{added} -{removed} gaussians, {len(gaussian_map)} total")

    if output:
        gaussian_map.export_ply(output)
        print(f"[mapper] saved {len(gaussian_map)} gaussians to {output}")
    viewer.stop()
