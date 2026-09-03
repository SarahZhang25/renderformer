"""Persistent-Blender batch renderer: ONE bpy process renders MANY scenes.

Kills the dominant CPU cost of the fleet (process launch + bpy import +
device init per scene, ~2-5s each). N workers share one scene-json pool via
an atomic-rename claim protocol, so any number of workers on any number of
GPUs can drain the same directory; a heartbeat file lets a supervisor detect
hung renders. Rendered jsons move to --claim_dir (the h5 converter's input),
EXRs land in --output_dir exactly like single-scene to_blend.py.

    CUDA_VISIBLE_DEVICES=3 BLENDER_BACKEND=OPTIX python to_blend_batch.py \
        --scene_dir .../s1tmp --output_dir .../s1tmp --claim_dir .../s1tmp/rendered \
        --resolution 256 --spp 256
"""
import argparse, glob, json, os, random, sys, tempfile, time, traceback

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)
# bpy MUST load before trimesh/pymeshlab: their embree builds otherwise win
# the symbol namespace and bpy's embree4 dies on rtcIsSYCLDeviceSupported.
import to_blend  # noqa: E402  (bpy + device init happen once, here)
from dacite import from_dict, Config  # noqa: E402
from scene_config import SceneConfig  # noqa: E402
from scene_mesh import generate_scene_mesh  # noqa: E402


def render_one(json_path, out_dir, resolution, spp):
    with open(json_path) as f:
        cfg = from_dict(data_class=SceneConfig, data=json.load(f),
                        config=Config(check_types=True, strict=True))
    base = os.path.basename(json_path)
    base = base[:base.index('.json')]
    output_base = os.path.join(out_dir, base)
    with tempfile.TemporaryDirectory() as td:
        mesh = os.path.join(td, "temp_mesh.obj")
        # Cache the built geometry beside the EXRs for the h5 converter to reuse.
        generate_scene_mesh(cfg, mesh, os.path.dirname(json_path),
                            geom_cache_path=output_base + '_geom.npz')
        to_blend.scene_to_img(
            scene_config=cfg, mesh_path=mesh, output_image_path=output_base,
            dump_blend_file=False, save_img=False,
            resolution=resolution, spp=spp, skip_rendering=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--claim_dir', required=True)
    ap.add_argument('--fail_dir', default=None)
    ap.add_argument('--resolution', type=int, default=512)
    ap.add_argument('--spp', type=int, default=512)
    ap.add_argument('--idle_exit_s', type=int, default=1800)
    ap.add_argument('--heartbeat', default=None)
    a = ap.parse_args()
    for d in (a.output_dir, a.claim_dir, a.fail_dir or a.claim_dir + "_failed"):
        os.makedirs(d, exist_ok=True)
    fail_dir = a.fail_dir or a.claim_dir + "_failed"
    tag = f".c{os.getpid()}"
    done = fails = 0
    idle_t0 = time.time()
    cands = []
    while True:
        if not cands:
            # One directory scan per ~thousands of claims: a 500k-entry dir
            # costs seconds per glob and 48 workers doing it per-scene was
            # the fleet's real bottleneck.
            cands = glob.glob(os.path.join(a.scene_dir, 'scene_*.json'))
            random.shuffle(cands)
            if not cands:
                if time.time() - idle_t0 > a.idle_exit_s:
                    break
                time.sleep(20)
                continue
        claimed = None
        while cands and not claimed:
            c = cands.pop()
            try:
                os.rename(c, c + tag)
                claimed = c + tag
            except OSError:
                continue
        if not claimed:
            continue
        idle_t0 = time.time()
        if a.heartbeat:
            open(a.heartbeat, "w").write(str(time.time()))
        orig = claimed[:-len(tag)]
        try:
            t0 = time.time()
            render_one(claimed, a.output_dir, a.resolution, a.spp)
            os.rename(claimed, os.path.join(a.claim_dir, os.path.basename(orig)))
            done += 1
            if done % 20 == 1:
                print(f"[batch] {done} scenes, last {time.time()-t0:.1f}s", flush=True)
        except Exception:
            fails += 1
            traceback.print_exc()
            try:
                os.rename(claimed, os.path.join(fail_dir, os.path.basename(orig)))
            except OSError:
                pass
            if fails > 200:
                print("[batch] too many failures, exiting", flush=True)
                break
    print(f"[batch] exit: done={done} fails={fails}", flush=True)


if __name__ == "__main__":
    main()
