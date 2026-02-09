import av
import cv2
import heapq
import torch
import clip
import imagehash
import numpy as np
import time
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from pathlib import Path


############################################
# GLOBAL MODEL LOAD (LOAD ONCE!)
############################################

print("Loading models...")

DEVICE = "cpu"

clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)

PROMPTS = clip.tokenize([
    "a clear readable document",
    "a sharp high quality photo",
    "blurry image",
    "low quality image"
]).to(DEVICE)

print("Models loaded ✅")

############################################
# SCENE DETECTION
############################################

def is_scene_changed(prev, curr, ssim_thresh=0.90, diff_thresh=8):

    prev_small = cv2.resize(prev, (320, 180))
    curr_small = cv2.resize(curr, (320, 180))

    prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)

    score = ssim(prev_gray, curr_gray)
    diff = cv2.absdiff(prev_gray, curr_gray)

    return score < ssim_thresh or np.mean(diff) > diff_thresh

def sharpness(img):
    return cv2.Laplacian(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        cv2.CV_64F
    ).var()


def quality_proxy(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    contrast = gray.std()
    noise = cv2.Laplacian(gray, cv2.CV_64F).var()

    contrast = min(contrast / 50 * 100, 100)
    noise = min(noise / 1000 * 100, 100)

    return contrast * 0.5 + noise * 0.5

def ocr_score(pil_img):
    img = np.array(pil_img.convert("L"))

    edges = cv2.Canny(img, 50, 150)

    edge_density = edges.mean()  # 0–255 range approx

    # Normalize to 0–100
    score = min(edge_density * 2, 100)

    return float(score)


def clip_score(pil_img):

    image = clip_preprocess(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits, _ = clip_model(image, PROMPTS)
        probs = logits.softmax(dim=-1)[0].cpu().numpy()

    score = (probs[0] + probs[1] - probs[2] - probs[3]) * 100

    return float(score)


############################################
# FRAME SCORER
############################################

class FrameScorer:

    def __init__(self):
        self.hashes = []

    def is_duplicate(self, pil):

        h = imagehash.phash(pil)

        for old in self.hashes:
            if abs(h - old) < 8:
                return True

        self.hashes.append(h)
        return False


    def score(self, img, sharp_thresh):

        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if self.is_duplicate(pil):
            return None

        sharp = sharpness(img)

        if sharp < sharp_thresh:
            return None

        resolution = min(
            (img.shape[0] * img.shape[1]) / 1e6 * 10,
            100
        )

        quality = quality_proxy(img)

        text_score = ocr_score(pil)

        qr_score = 0

        semantic = clip_score(pil)

        final = (
            semantic * 0.30 +
            sharp * 0.20 +
            resolution * 0.10 +
            quality * 0.10 +
            text_score * 0.25 +
            qr_score * 0.05
        )

        return final, img


############################################
# ZERO FAILURE PIPELINE
############################################

def extract_best_frames(video_path, top_k=5, frame_skip=5):

    start = time.time()

    print("\n=========== ZERO FAILURE PIPELINE ===========")
    print("Video:", video_path)

    def run_pass(sharp_thresh, ssim_thresh, allow_scene):

        container = av.open(video_path)
        stream = container.streams.video[0]
        scorer = FrameScorer()
        heap = []
        prev_frame = None
        decoded = 0
        accepted = 0

        for i, frame in enumerate(container.decode(stream)):

            decoded += 1

            if i % frame_skip != 0:
                continue

            img = frame.to_ndarray(format="bgr24")

            if allow_scene and prev_frame is not None:
                if not is_scene_changed(prev_frame, img, ssim_thresh):
                    prev_frame = img
                    continue

            prev_frame = img

            result = scorer.score(img, sharp_thresh)

            if result is None:
                continue

            score, image = result
            accepted += 1

            if len(heap) < top_k:
                heapq.heappush(heap, (score, image))
            else:
                if score > heap[0][0]:
                    heapq.heapreplace(heap, (score, image))

        container.close()

        print(f"Decoded: {decoded} | Accepted: {accepted}")

        heap.sort(key=lambda x: x[0], reverse=True)

        return [img for (_, img) in heap]


    ###################################
    # PASS 1 — Normal
    ###################################

    print("\nPASS 1 — Smart AI")

    frames = run_pass(
        sharp_thresh=25,
        ssim_thresh=0.92,
        allow_scene=True
    )

    if len(frames) >= top_k:
        print("PASS 1 SUCCESS ✅")
        print("Total time:", round(time.time() - start, 2), "sec")
        return frames


    ###################################
    # PASS 2 — Relaxed
    ###################################

    print("\nPASS 2 — Relaxing filters")

    frames = run_pass(
        sharp_thresh=10,
        ssim_thresh=0.85,
        allow_scene=False
    )

    if len(frames) > 0:
        print("PASS 2 SUCCESS ✅")
        print("Total time:", round(time.time() - start, 2), "sec")
        return frames


    ###################################
    # PASS 3 — Emergency sampler
    ###################################

    print("\nPASS 3 — Emergency fallback 🚨")

    container = av.open(video_path)
    stream = container.streams.video[0]

    emergency = []

    total_frames = int(stream.frames) if stream.frames else 300
    interval = max(1, total_frames // top_k)

    for i, frame in enumerate(container.decode(stream)):

        if i % interval == 0:
            img = frame.to_ndarray(format="bgr24")
            emergency.append(img)

        if len(emergency) >= top_k:
            break

    container.close()

    print("Emergency frames returned ✅")
    print("Total time:", round(time.time() - start, 2), "sec")

    return emergency


############################################
# SAVE OUTPUT
############################################

def save_frames(frames, output_folder="output_frames"):

    output = Path(output_folder)
    output.mkdir(exist_ok=True)

    for i, img in enumerate(frames):
        path = output / f"best_{i}.jpg"
        cv2.imwrite(str(path), img)

    print(f"\n✅ Saved {len(frames)} frames to '{output_folder}'")


############################################
# MAIN
############################################

# if __name__ == "__main__":

#     VIDEO_PATH = r"C:\Users\prath\Downloads\packaging.mp4"

#     frames = extract_best_frames(
#         VIDEO_PATH,
#         top_k=5,
#         frame_skip=5
#     )

#     print("Frames returned:", len(frames))

#     save_frames(frames)

