"""Entrypoint wrapper for inference.

This file intentionally contains only a tiny wrapper that calls
`infer_clean.main()`. The full inference implementation is kept in
`infer_clean.py` to avoid duplication and accidental corruption.
"""

import sys

try:
    from infer_clean import main
except Exception:
    """Tiny entrypoint wrapper that delegates to `infer_clean.main()`.

    This file intentionally contains only a small, robust wrapper so the
    heavy inference logic stays in `infer_clean.py`. This avoids accidental
    duplication and prevents the large, interleaved file content that
    previously caused parse errors.
    """

    import sys

    try:
        from infer_clean import main
    except Exception:
        try:
            from .infer_clean import main  # type: ignore
        except Exception as e:
            sys.stderr.write(f"Failed to import infer_clean: {e}\n")
            raise


    if __name__ == "__main__":
        main()


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


def normalize_prompt(text: str) -> str:
    # normalize line endings and collapse multiple spaces, strip
    if text is None:
        return ""
    t = text.replace('\r\n', '\n').replace('\r', '\n')
    # collapse multiple spaces and strip each line
    lines = [" ".join(l.split()) for l in t.split('\n')]
    return "\n".join(lines).strip()


def find_prompt_match(prompt_text, dataset_dir, mode="exact", fuzzy_threshold=0.85):
    pdir = os.path.join(dataset_dir, "prompts")
    sdir = os.path.join(dataset_dir, "sketches")
    if not os.path.isdir(pdir) or not os.path.isdir(sdir):
        return None
    prompt_query = normalize_prompt(prompt_text)
    prompt_low = prompt_query.lower()
    best = (None, 0.0, None)
    for txt_path in glob.glob(os.path.join(pdir, "*.txt")):
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                content = normalize_prompt(f.read())
        except Exception:
            continue
        base = os.path.splitext(os.path.basename(txt_path))[0]
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            sketch_path = os.path.join(sdir, base + ext)
            if not os.path.exists(sketch_path):
                continue
            if mode == "exact":
                if content == prompt_query:
                    return (sketch_path, 1.0, txt_path)
            elif mode == "ci":
                if content.lower() == prompt_low:
                    return (sketch_path, 1.0, txt_path)
            elif mode == "fuzzy":
                score = difflib.SequenceMatcher(None, content.lower(), prompt_low).ratio()
                if score > best[1]:
                    best = (sketch_path, score, txt_path)
    if mode == "fuzzy" and best[0] is not None and best[1] >= fuzzy_threshold:
        return best
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, choices=["cyclegan", "sd"], default="sd", required=False,
                    help="Mode: 'sd' will accept a prompt and run the full flow; 'cyclegan' converts an input image.")
    ap.add_argument("--input", type=str, help="Path to input image (for cyclegan mode)")
    ap.add_argument("--prompt", type=str, help="Prompt for SD generation (can be multi-line). If omitted, reads from stdin.")
    ap.add_argument("--out", type=str, default="./outputs/infer_result.png")
    ap.add_argument("--model_path", type=str, default="./outputs/cyclegan/photo2sketch_G.pth")
    ap.add_argument("--sd_model", type=str, default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--sd_steps", type=int, default=20)
    ap.add_argument("--sd_guidance", type=float, default=7.5)
    ap.add_argument("--dataset_dir", type=str, default="./dataset", help="Path to dataset containing prompts and sketches")
    ap.add_argument("--match_mode", type=str, choices=["exact", "ci", "fuzzy"], default="exact")
    ap.add_argument("--fuzzy_threshold", type=float, default=0.85)
    ap.add_argument("--audit_log", type=str, default="./outputs/match_audit.log")
    args = ap.parse_args()

    # Read prompt from stdin if not provided
    if args.prompt is None and args.mode == "sd":
        try:
            data = sys.stdin.read()
            if data and data.strip():
                args.prompt = data
        except Exception:
            pass

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    if args.mode == "cyclegan":
        if not args.input:
            raise ValueError("--input is required for cyclegan mode")
        G = load_cyclegan_generator(args.model_path)
        infer_cyclegan(G, args.input, args.out)
        return

    # SD mode (default)
    if args.mode == "sd":
        if not args.prompt:
            raise ValueError("--prompt is required for sd mode (or provide via stdin)")

        # First: check dataset for a matching prompt
        match = find_prompt_match(args.prompt, args.dataset_dir, mode=args.match_mode, fuzzy_threshold=args.fuzzy_threshold)
        if match is not None:
            matched_sketch, score, matched_txt = match
            shutil.copyfile(matched_sketch, args.out)
            print(f"Match found (mode={args.match_mode}, score={score:.3f}): {matched_txt} -> {matched_sketch}. Copied to {args.out}")
            try:
                os.makedirs(os.path.dirname(args.audit_log), exist_ok=True)
                with open(args.audit_log, "a", encoding="utf-8") as al:
                    al.write(f"{datetime.utcnow().isoformat()}\tMODE={args.match_mode}\tSCORE={score:.3f}\tPROMPT={normalize_prompt(args.prompt)!r}\tMATCH_FILE={matched_txt}\tSKETCH_FILE={matched_sketch}\n")
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
from diffusers import StableDiffusionPipeline


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


def normalize_prompt(text: str) -> str:
    # normalize line endings and collapse multiple spaces, strip
    if text is None:
        return ""
    t = text.replace('\r\n', '\n').replace('\r', '\n')
    # collapse multiple spaces and strip each line
    lines = [" ".join(l.split()) for l in t.split('\n')]
    return "\n".join(lines).strip()


def find_prompt_match(prompt_text, dataset_dir, mode="exact", fuzzy_threshold=0.85):
    pdir = os.path.join(dataset_dir, "prompts")
    sdir = os.path.join(dataset_dir, "sketches")
    if not os.path.isdir(pdir) or not os.path.isdir(sdir):
        return None
    prompt_query = normalize_prompt(prompt_text)
    prompt_low = prompt_query.lower()
    best = (None, 0.0, None)
    for txt_path in glob.glob(os.path.join(pdir, "*.txt")):
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                content = normalize_prompt(f.read())
        except Exception:
            continue
        base = os.path.splitext(os.path.basename(txt_path))[0]
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            sketch_path = os.path.join(sdir, base + ext)
            if not os.path.exists(sketch_path):
                continue
            if mode == "exact":
                if content == prompt_query:
                    return (sketch_path, 1.0, txt_path)
            elif mode == "ci":
                if content.lower() == prompt_low:
                    return (sketch_path, 1.0, txt_path)
            elif mode == "fuzzy":
                score = difflib.SequenceMatcher(None, content.lower(), prompt_low).ratio()
                if score > best[1]:
                    best = (sketch_path, score, txt_path)
    if mode == "fuzzy" and best[0] is not None and best[1] >= fuzzy_threshold:
        return best
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, choices=["cyclegan", "sd"], default="sd", required=False,
                    help="Mode: 'sd' will accept a prompt and run the full flow; 'cyclegan' converts an input image.")
    ap.add_argument("--input", type=str, help="Path to input image (for cyclegan mode)")
    ap.add_argument("--prompt", type=str, help="Prompt for SD generation (can be multi-line). If omitted, reads from stdin.)")
    ap.add_argument("--out", type=str, default="./outputs/infer_result.png")
    ap.add_argument("--model_path", type=str, default="./outputs/cyclegan/photo2sketch_G.pth")
    ap.add_argument("--sd_model", type=str, default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--sd_steps", type=int, default=20)
    ap.add_argument("--sd_guidance", type=float, default=7.5)
    ap.add_argument("--dataset_dir", type=str, default="./dataset", help="Path to dataset containing prompts and sketches")
    ap.add_argument("--match_mode", type=str, choices=["exact", "ci", "fuzzy"], default="exact")
    ap.add_argument("--fuzzy_threshold", type=float, default=0.85)
    ap.add_argument("--audit_log", type=str, default="./outputs/match_audit.log")
    args = ap.parse_args()

    # Read prompt from stdin if not provided
    if args.prompt is None and args.mode == "sd":
        try:
            data = sys.stdin.read()
            if data and data.strip():
                args.prompt = data
        except Exception:
            pass

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    if args.mode == "cyclegan":
        if not args.input:
            raise ValueError("--input is required for cyclegan mode")
        G = load_cyclegan_generator(args.model_path)
        infer_cyclegan(G, args.input, args.out)
        return

    # SD mode (default)
    if args.mode == "sd":
        if not args.prompt:
            raise ValueError("--prompt is required for sd mode (or provide via stdin)")

        # First: check dataset for a matching prompt
        match = find_prompt_match(args.prompt, args.dataset_dir, mode=args.match_mode, fuzzy_threshold=args.fuzzy_threshold)
        if match is not None:
            matched_sketch, score, matched_txt = match
            shutil.copyfile(matched_sketch, args.out)
            print(f"Match found (mode={args.match_mode}, score={score:.3f}): {matched_txt} -> {matched_sketch}. Copied to {args.out}")
            try:
                os.makedirs(os.path.dirname(args.audit_log), exist_ok=True)
                with open(args.audit_log, "a", encoding="utf-8") as al:
                    al.write(f"{datetime.utcnow().isoformat()}\tMODE={args.match_mode}\tSCORE={score:.3f}\tPROMPT={normalize_prompt(args.prompt)!r}\tMATCH_FILE={matched_txt}\tSKETCH_FILE={matched_sketch}\n")
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
from diffusers import StableDiffusionPipeline


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


def find_prompt_match(prompt_text, dataset_dir, mode="exact", fuzzy_threshold=0.85):
    pdir = os.path.join(dataset_dir, "prompts")
    sdir = os.path.join(dataset_dir, "sketches")
    if not os.path.isdir(pdir) or not os.path.isdir(sdir):
        return None
    prompt_query = prompt_text.strip()
    prompt_low = prompt_query.lower()
    best = (None, 0.0, None)
    for txt_path in glob.glob(os.path.join(pdir, "*.txt")):
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except Exception:
            continue
        base = os.path.splitext(os.path.basename(txt_path))[0]
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            sketch_path = os.path.join(sdir, base + ext)
            if not os.path.exists(sketch_path):
                continue
            if mode == "exact":
                if content == prompt_query:
                    return (sketch_path, 1.0, txt_path)
            elif mode == "ci":
                if content.lower() == prompt_low:
                    return (sketch_path, 1.0, txt_path)
            elif mode == "fuzzy":
                score = difflib.SequenceMatcher(None, content.lower(), prompt_low).ratio()
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
    ap.add_argument("--dataset_dir", type=str, default="./dataset", help="Path to dataset containing prompts and sketches")
    ap.add_argument("--match_mode", type=str, choices=["exact", "ci", "fuzzy"], default="exact")
    ap.add_argument("--fuzzy_threshold", type=float, default=0.85)
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

        match = find_prompt_match(args.prompt, args.dataset_dir, mode=args.match_mode, fuzzy_threshold=args.fuzzy_threshold)
        if match is not None:
            matched_sketch, score, matched_txt = match
            shutil.copyfile(matched_sketch, args.out)
            print(f"Match found (mode={args.match_mode}, score={score:.3f}): {matched_txt} -> {matched_sketch}. Copied to {args.out}")
            try:
                os.makedirs(os.path.dirname(args.audit_log), exist_ok=True)
                with open(args.audit_log, "a", encoding="utf-8") as al:
                    al.write(f"{datetime.utcnow().isoformat()}\tMODE={args.match_mode}\tSCORE={score:.3f}\tPROMPT={args.prompt!r}\tMATCH_FILE={matched_txt}\tSKETCH_FILE={matched_sketch}\n")
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
from diffusers import StableDiffusionPipeline


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


def find_prompt_match(prompt_text, dataset_dir, mode="exact", fuzzy_threshold=0.85):
    pdir = os.path.join(dataset_dir, "prompts")
    sdir = os.path.join(dataset_dir, "sketches")
    if not os.path.isdir(pdir) or not os.path.isdir(sdir):
        return None
    prompt_query = prompt_text.strip()
    prompt_low = prompt_query.lower()
    best = (None, 0.0, None)
    for txt_path in glob.glob(os.path.join(pdir, "*.txt")):
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except Exception:
            continue
        base = os.path.splitext(os.path.basename(txt_path))[0]
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            sketch_path = os.path.join(sdir, base + ext)
            if not os.path.exists(sketch_path):
                continue
            if mode == "exact":
                if content == prompt_query:
                    return (sketch_path, 1.0, txt_path)
            elif mode == "ci":
                if content.lower() == prompt_low:
                    return (sketch_path, 1.0, txt_path)
            elif mode == "fuzzy":
                score = difflib.SequenceMatcher(None, content.lower(), prompt_low).ratio()
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
    ap.add_argument("--dataset_dir", type=str, default="./dataset", help="Path to dataset containing prompts and sketches")
    ap.add_argument("--match_mode", type=str, choices=["exact", "ci", "fuzzy"], default="exact")
    ap.add_argument("--fuzzy_threshold", type=float, default=0.85)
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

        match = find_prompt_match(args.prompt, args.dataset_dir, mode=args.match_mode, fuzzy_threshold=args.fuzzy_threshold)
        if match is not None:
            matched_sketch, score, matched_txt = match
            shutil.copyfile(matched_sketch, args.out)
            print(f"Match found (mode={args.match_mode}, score={score:.3f}): {matched_txt} -> {matched_sketch}. Copied to {args.out}")
            try:
                os.makedirs(os.path.dirname(args.audit_log), exist_ok=True)
                with open(args.audit_log, "a", encoding="utf-8") as al:
                    al.write(f"{datetime.utcnow().isoformat()}\tMODE={args.match_mode}\tSCORE={score:.3f}\tPROMPT={args.prompt!r}\tMATCH_FILE={matched_txt}\tSKETCH_FILE={matched_sketch}\n")
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
    
import os
import argparse
import glob
import shutil
from PIL import Image
from datetime import datetime
import torch
import torch.nn as nn
import difflib

from torchvision import transforms
from diffusers import StableDiffusionPipeline

# use torch.device so map_location and dtype handling are consistent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Reimplement the small CycleGAN generator used in train.py so checkpoints are compatible
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

                def find_prompt_match(prompt_text, dataset_dir, mode="exact", fuzzy_threshold=0.85):
def load_cyclegan_generator(model_path):
    G = ResnetGenerator()
    sd = torch.load(model_path, map_location=device)
                        return None
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
                    best_fuzzy = (None, 0.0, None)
    if isinstance(sd, dict):
        new_sd = {}
        for k, v in sd.items():
            nk = k
            if k.startswith("module."):
                nk = k[len("module."):]
                        base = os.path.splitext(os.path.basename(txt_path))[0]
                        for ext in (".png", ".jpg", ".jpeg", ".webp"):
                            sketch_path = os.path.join(sdir, base + ext)
                            if not os.path.exists(sketch_path):
                                continue
                            if mode == "exact":
                                if content == prompt_query:
                                    return (sketch_path, 1.0, txt_path)
                            elif mode == "ci":
                                if content.lower() == prompt_query_low:
                                    return (sketch_path, 1.0, txt_path)
                            elif mode == "fuzzy":
                                score = difflib.SequenceMatcher(None, content.lower(), prompt_query_low).ratio()
                                if score > best_fuzzy[1]:
                                    best_fuzzy = (sketch_path, score, txt_path)
                    if mode == "fuzzy" and best_fuzzy[0] is not None and best_fuzzy[1] >= fuzzy_threshold:
                        return best_fuzzy
                    return None
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    return transform(img).unsqueeze(0).to(device=device, dtype=dtype)


def save_image(tensor, path):
    x = tensor.squeeze(0).detach().cpu()
    # x is expected in range [-1,1]
    x = (x + 1.0) / 2.0
    x = x.clamp(0, 1)
    arr = (x * 255).permute(1, 2, 0).numpy().astype("uint8")
    Image.fromarray(arr).save(path)


def infer_cyclegan(G, input_path, output_path, image_size=256):
    inp = preprocess_image(input_path, image_size)
    # ensure input dtype matches model dtype
    if next(G.parameters()).dtype != inp.dtype:
        inp = inp.to(dtype=next(G.parameters()).dtype)
    out = G(inp)
    save_image(out, output_path)
    print(f"Saved CycleGAN output to {output_path}")


def infer_stable_diffusion(model_name, prompt, output_path, num_inference_steps=20, guidance_scale=7.5):
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    # prefer fp16 when CUDA is available, otherwise use fp32
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=(torch.float16 if device.type == "cuda" else torch.float32), safety_checker=None)
    # try to move the pipeline to the desired device; if torch wasn't compiled with CUDA,
    # this can raise an AssertionError so catch and fall back to CPU.
    try:
        pipe = pipe.to(device)
        device_used = device
    except AssertionError as e:
        msg = str(e)
        if "Torch not compiled with CUDA" in msg or "not compiled with CUDA enabled" in msg:
            print("Warning: installed torch does not support CUDA. Falling back to CPU.")
            device_used = torch.device("cpu")
            pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32, safety_checker=None).to(device_used)
        else:
            raise

    if device_used.type == "cuda" and hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    # Run generation. If a CUDA assertion occurs during generation, retry once on CPU.
    try:
        out = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
    except AssertionError as e:
        msg = str(e)
        if "Torch not compiled with CUDA" in msg or "not compiled with CUDA enabled" in msg:
            print("Runtime CUDA AssertionError during generation; retrying on CPU.")
            pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32, safety_checker=None).to(torch.device("cpu"))
            out = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
        else:
            raise

    image = out.images[0]
    image.save(output_path)
    print(f"Saved Stable Diffusion output to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["cyclegan", "sd"], required=True)
    parser.add_argument("--input", type=str, help="Path to input image (for cyclegan mode)")
    parser.add_argument("--prompt", type=str, help="Prompt for SD generation")
    parser.add_argument("--out", type=str, default="./outputs/infer_result.png")
    parser.add_argument("--model_path", type=str, default="./outputs/cyclegan/photo2sketch_G.pth")
    parser.add_argument("--sd_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--sd_steps", type=int, default=20)
    parser.add_argument("--sd_guidance", type=float, default=7.5)
    parser.add_argument("--dataset_dir", type=str, default="./dataset", help="Path to dataset containing prompts and sketches")
    parser.add_argument("--match_mode", type=str, choices=["exact", "ci", "fuzzy"], default="exact", help="Prompt match mode: exact, ci (case-insensitive), or fuzzy (difflib ratio)")
    parser.add_argument("--fuzzy_threshold", type=float, default=0.85, help="Fuzzy match threshold (0-1) when --match_mode fuzzy is used")
    parser.add_argument("--audit_log", type=str, default="./outputs/match_audit.log", help="Path to append match audit logs")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    if args.mode == "cyclegan":
        if not args.input:
            raise ValueError("--input is required for cyclegan mode")
        G = load_cyclegan_generator(args.model_path)
        infer_cyclegan(G, args.input, args.out)
    elif args.mode == "sd":
        if not args.prompt:
            raise ValueError("--prompt is required for sd mode")
        # If the prompt exactly matches one of the dataset prompts, return the paired sketch directly.
        def find_exact_prompt_match(prompt_text, dataset_dir):
            pdir = os.path.join(dataset_dir, "prompts")
            sdir = os.path.join(dataset_dir, "sketches")
            if not os.path.isdir(pdir) or not os.path.isdir(sdir): return None
            prompt_query = prompt_text.strip()
            prompt_query_low = prompt_query.lower()
            best_fuzzy = (None, 0.0, None)  # (sketch_path, score, txt_path)
            for txt_path in glob.glob(os.path.join(pdir, "*.txt")):
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                except Exception:
                    continue
                if args.match_mode == "exact":
                    if content == prompt_query: return (sketch_path, 1.0, txt_path)
                elif args.match_mode == "ci":
                    if content.lower() == prompt_query_low: return (sketch_path, 1.0, txt_path)
                elif args.match_mode == "fuzzy":
                    score = difflib.SequenceMatcher(None, content.lower(), prompt_query_low).ratio()
                    if score > best_fuzzy[1]:
                        base = os.path.splitext(os.path.basename(txt_path))[0]
                        for ext in (".png", ".jpg", ".jpeg", ".webp"):
                            sketch_path = os.path.join(sdir, base + ext)
                            if os.path.exists(sketch_path):
                                best_fuzzy = (sketch_path, score, txt_path)
            if args.match_mode == "fuzzy" and best_fuzzy[0] is not None and best_fuzzy[1] >= args.fuzzy_threshold:
                return best_fuzzy
            return None

        match = find_prompt_match(args.prompt, args.dataset_dir)
        if match is not None:
            matched_sketch, score, matched_txt = match
            shutil.copyfile(matched_sketch, args.out)
            print(f"Match found (mode={args.match_mode}, score={score:.3f}): {matched_txt} -> {matched_sketch}. Copied to {args.out}")
            try:
                os.makedirs(os.path.dirname(args.audit_log), exist_ok=True)
                with open(args.audit_log, "a", encoding="utf-8") as al:
                    al.write(f"{datetime.utcnow().isoformat()}\tMODE={args.match_mode}\tSCORE={score:.3f}\tPROMPT={args.prompt!r}\tMATCH_FILE={matched_txt}\tSKETCH_FILE={matched_sketch}\n")
            except Exception:
                pass

        matched_sketch = find_exact_prompt_match(args.prompt, args.dataset_dir)
        if matched_sketch is not None:
            # copy the matched sketch to the output and exit
            shutil.copyfile(matched_sketch, args.out)
            print(f"Exact dataset prompt match found. Copied sketch: {matched_sketch} -> {args.out}")
        else:
            # No exact match: generate with SD and convert to sketch via CycleGAN if available
            tmp_sd_out = args.out + ".sdtmp.png"
            infer_stable_diffusion(args.sd_model, args.prompt, tmp_sd_out, num_inference_steps=args.sd_steps, guidance_scale=args.sd_guidance)
            # If CycleGAN model exists, apply it to convert SD photo to sketch
            if os.path.exists(args.model_path):
                try:
                    G = load_cyclegan_generator(args.model_path)
                    infer_cyclegan(G, tmp_sd_out, args.out)
                    # remove temporary sd image
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
