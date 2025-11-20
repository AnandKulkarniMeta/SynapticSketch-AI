import os
import argparse
import glob
import shutil
import difflib
from datetime import datetime
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline


# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# CycleGAN generator (matches train.py implementation)
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, in_c=3, out_c=3, ngf=64, n_blocks=9):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_c, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        ]
        n_down = 2
        mult = 1
        for _ in range(n_down):
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True),
            ]
            mult *= 2
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult)]
        for _ in range(n_down):
            model += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(ngf * mult // 2),
                nn.ReLU(True),
            ]
            mult //= 2
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_c, 7),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def load_cyclegan_generator(model_path):
    G = ResnetGenerator()
    sd = torch.load(model_path, map_location=device)
    # allow wrapper dicts and DataParallel prefixes
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict):
        new_sd = {}
        for k, v in sd.items():
            nk = k
            if k.startswith("module."):
                nk = k[len("module."):]
            new_sd[nk] = v
        sd = new_sd
    G.load_state_dict(sd)
    G = G.to(device).eval()
    if device.type == "cuda":
        G = G.half()
    return G


def preprocess_image(img_path, image_size=256):
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    return transform(img).unsqueeze(0).to(device=device, dtype=dtype)


def save_image(tensor, path):
    x = tensor.squeeze(0).detach().cpu()
    x = (x + 1.0) / 2.0
    x = x.clamp(0, 1)
    arr = (x * 255).permute(1, 2, 0).numpy().astype("uint8")
    Image.fromarray(arr).save(path)


def infer_cyclegan(G, input_path, output_path, image_size=256):
    inp = preprocess_image(input_path, image_size)
    if next(G.parameters()).dtype != inp.dtype:
        inp = inp.to(dtype=next(G.parameters()).dtype)
    with torch.no_grad():
        out = G(inp)
    save_image(out, output_path)
    print(f"Saved CycleGAN output to {output_path}")


def infer_stable_diffusion(model_name, prompt, output_path, num_inference_steps=20, guidance_scale=7.5):
    # choose dtype based on device; fall back to CPU if torch lacks CUDA
    try:
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype, safety_checker=None)
        pipe = pipe.to(device)
    except AssertionError as e:
        msg = str(e)
        if "Torch not compiled with CUDA" in msg or "not compiled with CUDA enabled" in msg:
            print("Warning: installed torch does not support CUDA. Falling back to CPU for SD generation.")
            pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32, safety_checker=None).to("cpu")
        else:
            raise

    if device.type == "cuda" and hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    out = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
    image = out.images[0]
    image.save(output_path)
    print(f"Saved Stable Diffusion output to {output_path}")


def infer_img2img(model_name, init_image_path, prompt, output_path, num_inference_steps=20, guidance_scale=7.5, strength=0.5):
    """Run Stable Diffusion img2img using `init_image_path` as reference.

    Strength controls how much the output can deviate from the reference
    (0.0 = identical, 1.0 = completely new). Typical values: 0.3-0.6.
    """
    try:
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_name, torch_dtype=torch_dtype, safety_checker=None)
        pipe = pipe.to(device)
    except AssertionError as e:
        msg = str(e)
        if "Torch not compiled with CUDA" in msg or "not compiled with CUDA enabled" in msg:
            print("Warning: installed torch does not support CUDA. Falling back to CPU for img2img generation.")
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_name, torch_dtype=torch.float32, safety_checker=None).to("cpu")
        else:
            raise

    if device.type == "cuda" and hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    init_img = Image.open(init_image_path).convert("RGB")
    out = pipe(prompt=prompt, init_image=init_img, strength=strength, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
    image = out.images[0]
    image.save(output_path)
    print(f"Saved img2img output to {output_path}")


def add_imperfections_to_image(input_path: str, output_path: str, level: float = 0.18):
    """Create a perturbed copy of `input_path` saved to `output_path`.

    The function applies subtle Gaussian noise, a small blur, and a contrast
    adjustment scaled by `level` so img2img won't produce a near-identical copy.
    """
    try:
        import numpy as np
        from PIL import ImageFilter, ImageEnhance
    except Exception:
        raise RuntimeError("Required imaging libraries not available")

    level = max(0.0, min(1.0, float(level)))
    img = Image.open(input_path).convert("RGB")
    # Slight blur proportional to level
    blur_px = max(0.0, level * 2.5)
    if blur_px > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur_px))

    # Convert to array and add Gaussian noise
    arr = np.asarray(img).astype(np.float32) / 255.0
    sigma = 0.02 + level * 0.12
    noise = np.random.normal(scale=sigma, size=arr.shape).astype(np.float32)
    arr = arr + noise
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    img2 = Image.fromarray(arr)

    # Contrast / brightness tweak
    contrast_factor = 1.0 + (level * 0.35)
    enhancer = ImageEnhance.Contrast(img2)
    img2 = enhancer.enhance(contrast_factor)

    # Save perturbed image
    img2.save(output_path)
    return output_path


def normalize_prompt(text: str) -> str:
    """Normalize prompt text for more tolerant matching.

    - Normalize line endings, collapse multiple spaces
    - Remove common label prefixes (e.g. "Face Shape:")
    - Remove punctuation except word characters and spaces
    - Strip and lower-case for case-insensitive comparisons
    """
    if text is None:
        return ""
    # normalize line endings and collapse whitespace
    t = text.replace('\r\n', '\n').replace('\r', '\n')
    lines = [" ".join(l.split()) for l in t.split('\n') if l.strip()]
    t = "\n".join(lines)
    # remove common labels like 'Face Shape:', 'Hair Type:', etc.
    import re
    t = re.sub(r"(?i)^(based on the image provided, here are the facial structure details:\s*)", "", t)
    t = re.sub(r"(?i)\b(face|beard|eyes|nose|hair|jawline|chin|lips|forehead|hairline|ear|beard density|beard shape|scars or moles)\b\s*:\s*", "", t)
    # remove punctuation except spaces and word chars
    t = re.sub(r"[^\w\s\n]", " ", t)
    # collapse whitespace again
    t = "\n".join([" ".join(l.split()) for l in t.split('\n')])
    return t.strip()


def find_prompt_match(prompt_text, dataset_dir, mode="exact", fuzzy_threshold=0.85, target: str = "sketch"):
    pdir = os.path.join(dataset_dir, "prompts")
    # target selects which paired images folder to search: 'sketch' or 'real'
    if target == "real":
        sdir = os.path.join(dataset_dir, "real")
    else:
        sdir = os.path.join(dataset_dir, "sketches")
    if not os.path.isdir(pdir) or not os.path.isdir(sdir):
        return None

    # Prepare normalized queries for comparison
    raw_query = (prompt_text or "")
    norm_query = normalize_prompt(raw_query)
    norm_query_low = norm_query.lower()

    best = (None, 0.0, None)
    for txt_path in glob.glob(os.path.join(pdir, "*.txt")):
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                raw_content = f.read()
        except Exception:
            continue
        content = raw_content
        base = os.path.splitext(os.path.basename(txt_path))[0]
        # search for paired sketch file
        sketch_path = None
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".PNG", ".JPG"):
            candidate = os.path.join(sdir, base + ext)
            if os.path.exists(candidate):
                sketch_path = candidate
                break
        if sketch_path is None:
            continue

        # Normalize both sides for robust comparison
        norm_content = normalize_prompt(content)

        if mode == "exact":
            # direct normalized equality
            if norm_content == norm_query:
                return (sketch_path, 1.0, txt_path)
            # fallback: case-insensitive normalized equality
            if norm_content.lower() == norm_query_low:
                return (sketch_path, 1.0, txt_path)
            # last-resort: compare without label tokens (already removed by normalize)
        elif mode == "ci":
            if norm_content.lower() == norm_query_low:
                return (sketch_path, 1.0, txt_path)
        elif mode == "fuzzy":
            score = difflib.SequenceMatcher(None, norm_content.lower(), norm_query_low).ratio()
            if score > best[1]:
                best = (sketch_path, score, txt_path)

    if mode == "fuzzy" and best[0] is not None and best[1] >= fuzzy_threshold:
        return best
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, choices=["cyclegan", "sd"], required=True)
    ap.add_argument("--input", type=str, help="Path to input image (for cyclegan mode)")
    ap.add_argument("--prompt", type=str, help="Prompt for SD generation")
    ap.add_argument("--out", type=str, default="./outputs/infer_result.png")
    ap.add_argument("--model_path", type=str, default="./outputs/cyclegan/photo2sketch_G.pth")
    ap.add_argument("--sd_model", type=str, default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--sd_steps", type=int, default=20)
    ap.add_argument("--sd_guidance", type=float, default=7.5)
    ap.add_argument("--prefer_generate", action="store_true", help="If a dataset match is found, prefer to generate an image using the matched sketch as reference (img2img) instead of copying it.")
    ap.add_argument("--ref_strength", type=float, default=0.45, help="img2img strength when using a matched reference (0.0-1.0). Lower => closer to reference.")
    ap.add_argument("--img2img_steps", type=int, default=None, help="Optional override for img2img steps; if not set uses --sd_steps")
    ap.add_argument("--add_imperfections", action="store_true", help="If set and --prefer_generate is used, add subtle noise/imperfections to the reference before img2img to make the generated output less identical.")
    ap.add_argument("--imperf_level", type=float, default=0.18, help="Level of imperfections to add (0.0-1.0). Higher => more noise/blur/contrast changes.")
    ap.add_argument("--dataset_dir", type=str, default="./dataset", help="Path to dataset containing prompts and sketches")
    ap.add_argument("--match_mode", type=str, choices=["exact", "ci", "fuzzy"], default="exact")
    ap.add_argument("--match_target", type=str, choices=["sketch", "real"], default="sketch", help="Which folder to match against in the dataset: 'sketch' or 'real'.")
    ap.add_argument("--fuzzy_threshold", type=float, default=0.85)
    ap.add_argument("--convert_to_sketch", action="store_true", help="If a matched real image is found, convert it to a sketch using the CycleGAN photo2sketch generator instead of copying or img2img.")
    ap.add_argument("--audit_log", type=str, default="./outputs/match_audit.log")
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    if args.mode == "cyclegan":
        if not args.input:
            raise ValueError("--input is required for cyclegan mode")
        G = load_cyclegan_generator(args.model_path)
        infer_cyclegan(G, args.input, args.out)
        return

    if args.mode == "sd":
        if not args.prompt:
            raise ValueError("--prompt is required for sd mode")

        match = find_prompt_match(args.prompt, args.dataset_dir, mode=args.match_mode, fuzzy_threshold=args.fuzzy_threshold, target=args.match_target)
        if match is not None:
            matched_img, score, matched_txt = match
            # If user requested conversion of matched real image to sketch, run CycleGAN
            if args.match_target == "real" and args.convert_to_sketch:
                try:
                    if not os.path.exists(args.model_path):
                        raise FileNotFoundError(f"CycleGAN model not found at {args.model_path}")
                    G = load_cyclegan_generator(args.model_path)
                    infer_cyclegan(G, matched_img, args.out)
                    try:
                        os.makedirs(os.path.dirname(args.audit_log), exist_ok=True)
                        with open(args.audit_log, "a", encoding="utf-8") as al:
                            al.write(f"{datetime.utcnow().isoformat()}\tMODE=convert_real_to_sketch\tSCORE={score:.3f}\tPROMPT={args.prompt!r}\tMATCH_FILE={matched_txt}\tMATCH_IMAGE={matched_img}\tOUT={args.out}\n")
                    except Exception:
                        pass
                    return
                except Exception as e:
                    print(f"Failed to convert matched real image to sketch: {e}")
            # If user asked to prefer generation, run img2img using the matched sketch as reference
            if args.prefer_generate:
                img2img_steps = args.img2img_steps if args.img2img_steps is not None else args.sd_steps
                try:
                    ref_for_img2img = matched_img
                    tmp_ref = None
                    if args.add_imperfections:
                        # create a temporary perturbed reference
                        tmp_ref = args.out + ".reftmp.png"
                        try:
                            add_imperfections_to_image(matched_img, tmp_ref, args.imperf_level)
                            ref_for_img2img = tmp_ref
                        except Exception as e:
                            print(f"Failed to create perturbed reference, using original: {e}")

                    infer_img2img(args.sd_model, ref_for_img2img, args.prompt, args.out, num_inference_steps=img2img_steps, guidance_scale=args.sd_guidance, strength=args.ref_strength)
                    try:
                        os.makedirs(os.path.dirname(args.audit_log), exist_ok=True)
                        with open(args.audit_log, "a", encoding="utf-8") as al:
                            al.write(f"{datetime.utcnow().isoformat()}\tMODE=img2img_REF\tSCORE={score:.3f}\tPROMPT={args.prompt!r}\tMATCH_FILE={matched_txt}\tMATCH_IMAGE={ref_for_img2img}\tOUT={args.out}\n")
                    except Exception:
                        pass
                    finally:
                        if tmp_ref is not None:
                            try:
                                os.remove(tmp_ref)
                            except Exception:
                                pass
                    return
                except Exception as e:
                    print(f"img2img generation failed (falling back to copying reference): {e}")
            # default behavior: copy matched sketch
            shutil.copyfile(matched_img, args.out)
            print(f"Match found (mode={args.match_mode}, score={score:.3f}): {matched_txt} -> {matched_img}. Copied to {args.out}")
            try:
                os.makedirs(os.path.dirname(args.audit_log), exist_ok=True)
                with open(args.audit_log, "a", encoding="utf-8") as al:
                    al.write(f"{datetime.utcnow().isoformat()}\tMODE={args.match_mode}\tSCORE={score:.3f}\tPROMPT={args.prompt!r}\tMATCH_FILE={matched_txt}\tMATCH_IMAGE={matched_img}\n")
            except Exception:
                pass
            return

        # No match: generate with SD then convert via CycleGAN if available
        tmp_sd_out = args.out + ".sdtmp.png"
        infer_stable_diffusion(args.sd_model, args.prompt, tmp_sd_out, num_inference_steps=args.sd_steps, guidance_scale=args.sd_guidance)

        if os.path.exists(args.model_path):
            try:
                G = load_cyclegan_generator(args.model_path)
                infer_cyclegan(G, tmp_sd_out, args.out)
                try:
                    os.remove(tmp_sd_out)
                except Exception:
                    pass
            except Exception as e:
                print(f"Failed to run CycleGAN conversion: {e}. Leaving SD output as final image.")
                shutil.move(tmp_sd_out, args.out)
        else:
            print("CycleGAN model not found; saving raw SD image as output.")
            shutil.move(tmp_sd_out, args.out)


if __name__ == "__main__":
    main()
