"""Dense optical flow (Farneback): computation, quantized JPEG storage, visualization."""
import cv2
import numpy as np

# Farneback parameters from the assignment (Section 5.3)
FARNEBACK = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                 poly_n=5, poly_sigma=1.2, flags=0)


def read_gray(video_path, wanted, size):
    """Sequentially decode the wanted frame indices as resized grayscale. size=(w, h)."""
    want, got = set(int(i) for i in wanted), {}
    cap, i = cv2.VideoCapture(video_path), 0
    while len(got) < len(want) and cap.grab():
        if i in want:
            ok, f = cap.retrieve()
            if ok:
                got[i] = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), size,
                                    interpolation=cv2.INTER_AREA)
        i += 1
    cap.release()
    return got


def compute_flows(video_path, src_idx, size):
    """Flow from frame j to j+1 for every j in src_idx -> ((K, h, w, 2) float32, #zero-filled).

    At the last frame of a clip, (j-1 -> j) is used instead. If neither pair is
    decodable, the flow is zero-filled and counted.
    """
    wanted = set()
    for j in src_idx:
        wanted |= {max(int(j) - 1, 0), int(j), int(j) + 1}
    got = read_gray(video_path, wanted, size)
    flows, n_zero = [], 0
    for j in src_idx:
        j = int(j)
        for a, b in ((j, j + 1), (j - 1, j)):
            if a in got and b in got and a != b:
                flows.append(cv2.calcOpticalFlowFarneback(got[a], got[b], None, **FARNEBACK))
                break
        else:
            flows.append(np.zeros((size[1], size[0], 2), np.float32))
            n_zero += 1
    return np.stack(flows), n_zero


def quantize_flow(flow, bound):
    """px -> uint8. 0 px maps exactly to 128; +-bound px maps to 255 / 1 (values beyond are clipped)."""
    return np.clip(np.round(flow / bound * 127) + 128, 1, 255).astype(np.uint8)


def dequantize_flow(q):
    """uint8 -> float32 in [-1, 1], in units of the bound (multiply by bound for pixels)."""
    return (q.astype(np.float32) - 128.0) / 127.0


def save_flow_npz(path, flows, bound, src_idx, jpeg_q=95):
    """Store K flow fields as 2K grayscale JPEGs (u0, v0, u1, v1, ...) in one .npz."""
    q = quantize_flow(flows, bound)
    jpgs = [cv2.imencode('.jpg', np.ascontiguousarray(q[k, :, :, c]),
                         [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])[1].ravel()
            for k in range(len(q)) for c in range(2)]
    offsets = np.concatenate([[0], np.cumsum([len(j) for j in jpgs])])
    with open(path, 'wb') as fh:
        np.savez(fh, data=np.concatenate(jpgs), offsets=offsets,
                 src_idx=np.asarray(src_idx), bound=np.float32(bound))


def decode_flow(data, offsets, which):
    """Decode selected flow frames -> (n, h, w, 2) float32 in [-1, 1]."""
    out = []
    for k in which:
        u = cv2.imdecode(data[offsets[2 * k]:offsets[2 * k + 1]], cv2.IMREAD_GRAYSCALE)
        v = cv2.imdecode(data[offsets[2 * k + 1]:offsets[2 * k + 2]], cv2.IMREAD_GRAYSCALE)
        out.append(np.stack([u, v], axis=-1))
    return dequantize_flow(np.stack(out))


def load_flow(path, which=None):
    z = np.load(path)
    data, off = z['data'], z['offsets']
    k = (len(off) - 1) // 2
    return decode_flow(data, off, range(k) if which is None else which), float(z['bound'])


def flow_to_hsv(flow, max_mag=None):
    """(h, w, 2) flow -> RGB image: hue = direction, brightness = speed.

    Pass a shared max_mag when comparing several frames or clips; per-image
    scaling makes a nearly static frame look as 'bright' as a fast one.
    """
    u = np.ascontiguousarray(flow[..., 0], np.float32)
    v = np.ascontiguousarray(flow[..., 1], np.float32)
    mag, ang = cv2.cartToPolar(u, v)
    hsv = np.zeros((*flow.shape[:2], 3), np.uint8)
    hsv[..., 0] = (ang * 90 / np.pi).astype(np.uint8)          # OpenCV hue range is 0..179
    hsv[..., 1] = 255
    m = max_mag if max_mag else mag.max() + 1e-6
    hsv[..., 2] = np.clip(mag / m * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def color_wheel(size=101):
    """Legend for flow_to_hsv: the color of each motion direction."""
    yy, xx = np.mgrid[-1:1:size * 1j, -1:1:size * 1j].astype(np.float32)
    img = flow_to_hsv(np.stack([xx, yy], -1), max_mag=1.0)
    img[(xx ** 2 + yy ** 2) > 1] = 255
    return img
