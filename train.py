    import os
    import argparse
    from glob import glob

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T
    from PIL import Image
    from tqdm import tqdm

    from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
    from transformers import CLIPTokenizer, CLIPTextModel

    class PairedFacesDataset(Dataset):
        def __init__(self, root, image_size=256):
            self.real_dir = os.path.join(root, "real")
            self.sketch_dir = os.path.join(root, "sketches")
            self.prompt_dir = os.path.join(root, "prompts")
            assert os.path.isdir(self.real_dir), f"Missing {self.real_dir}"
            assert os.path.isdir(self.sketch_dir), f"Missing {self.sketch_dir}"
            assert os.path.isdir(self.prompt_dir), f"Missing {self.prompt_dir}"
            real_imgs = sorted(glob(os.path.join(self.real_dir, "*.*")))
            names = [os.path.splitext(os.path.basename(p))[0] for p in real_imgs]
            self.items = []
            for n in names:
                r = os.path.join(self.real_dir, n + _ext_exists(self.real_dir, n))
                s = os.path.join(self.sketch_dir, n + _ext_exists(self.sketch_dir, n))
                t = os.path.join(self.prompt_dir, n + ".txt")
                if os.path.exists(r) and os.path.exists(s) and os.path.exists(t):
                    self.items.append((r, s, t))
            if len(self.items) == 0:
                raise RuntimeError("No paired (real, sketch, prompt) found. Check names and folders.")
            self.transform = T.Compose([
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize([0.5]*3, [0.5]*3),
            ])

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            real_path, sketch_path, txt_path = self.items[idx]
            real_img = Image.open(real_path).convert("RGB")
            sketch_img = Image.open(sketch_path).convert("RGB")
            with open(txt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            return {
                "real": self.transform(real_img),
                "sketch": self.transform(sketch_img),
                "prompt": prompt
            }

    def _ext_exists(folder, name_no_ext):
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            if os.path.exists(os.path.join(folder, name_no_ext + ext)):
                return ext
        return ".png"

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
                nn.ReLU(True)
            ]
            n_down = 2
            mult = 1
            for i in range(n_down):
                model += [
                    nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(ngf * mult * 2),
                    nn.ReLU(True)
                ]
                mult *= 2
            for _ in range(n_blocks):
                model += [ResnetBlock(ngf * mult)]
            for i in range(n_down):
                model += [
                    nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, 3, stride=2, padding=1, output_padding=1),
                    nn.InstanceNorm2d(ngf * mult // 2),
                    nn.ReLU(True)
                ]
                mult //= 2
            model += [
                nn.ReflectionPad2d(3),
                nn.Conv2d(ngf, out_c, 7),
                nn.Tanh()
            ]
            self.model = nn.Sequential(*model)

        def forward(self, x):
            return self.model(x)

    class PatchDiscriminator(nn.Module):
        def __init__(self, in_c=3, ndf=64, n_layers=3):
            super().__init__()
            kw = 4; padw = 1
            seq = [nn.Conv2d(in_c, ndf, kw, 2, padw), nn.LeakyReLU(0.2, True)]
            nf_mult = 1
            for n in range(1, n_layers):
                nf_mult_prev = nf_mult
                nf_mult = min(2**n, 8)
                seq += [
                    nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kw, 2, padw),
                    nn.InstanceNorm2d(ndf*nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            seq += [nn.Conv2d(ndf*nf_mult, 1, kw, 1, padw)]
            self.model = nn.Sequential(*seq)

        def forward(self, x):
            return self.model(x)

    class ImagePool:
        def __init__(self, pool_size=50):
            self.pool_size = pool_size
            self.images = []

        def query(self, imgs):
            if self.pool_size == 0:
                return imgs
            result = []
            for img in imgs:
                img = img.detach()
                if len(self.images) < self.pool_size:
                    self.images.append(img)
                    result.append(img)
                else:
                    if torch.rand(1).item() > 0.5:
                        idx = torch.randint(0, self.pool_size, (1,)).item()
                        tmp = self.images[idx].clone()
                        self.images[idx] = img
                        result.append(tmp)
                    else:
                        result.append(img)
            return torch.stack(result, dim=0)

    def train_cyclegan(dataloader, device, epochs=100, lr=2e-4, out_dir="./outputs/cyclegan"):
        os.makedirs(out_dir, exist_ok=True)
        G_A2B = ResnetGenerator().to(device)
        G_B2A = ResnetGenerator().to(device)
        D_A = PatchDiscriminator().to(device)
        D_B = PatchDiscriminator().to(device)
        g_opt = optim.Adam(list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=lr, betas=(0.5, 0.999))
        d_opt = optim.Adam(list(D_A.parameters()) + list(D_B.parameters()), lr=lr, betas=(0.5, 0.999))
        adv = nn.MSELoss()
        cycle = nn.L1Loss()
        identity = nn.L1Loss()
        pool_A = ImagePool(50)
        pool_B = ImagePool(50)
        for epoch in range(1, epochs+1):
            pbar = tqdm(dataloader, desc=f"[CycleGAN] Epoch {epoch}/{epochs}")
            for batch in pbar:
                real_A = batch["real"].to(device)
                real_B = batch["sketch"].to(device)
                g_opt.zero_grad()
                idt_B = G_A2B(real_B)
                idt_A = G_B2A(real_A)
                loss_idt = identity(idt_A, real_A)*5.0 + identity(idt_B, real_B)*5.0
                fake_B = G_A2B(real_A)
                pred_fake_B = D_B(fake_B)
                loss_G_A2B = adv(pred_fake_B, torch.ones_like(pred_fake_B))
                fake_A = G_B2A(real_B)
                pred_fake_A = D_A(fake_A)
                loss_G_B2A = adv(pred_fake_A, torch.ones_like(pred_fake_A))
                rec_A = G_B2A(fake_B)
                rec_B = G_A2B(fake_A)
                loss_cycle = cycle(rec_A, real_A)*10.0 + cycle(rec_B, real_B)*10.0
                loss_G = loss_idt + loss_G_A2B + loss_G_B2A + loss_cycle
                loss_G.backward()
                g_opt.step()
                d_opt.zero_grad()
                pred_real_A = D_A(real_A)
                loss_D_A_real = adv(pred_real_A, torch.ones_like(pred_real_A))
                fake_A_pool = pool_A.query([fa for fa in fake_A])
                pred_fake_A = D_A(fake_A_pool.detach())
                loss_D_A_fake = adv(pred_fake_A, torch.zeros_like(pred_fake_A))
                loss_D_A = (loss_D_A_real + loss_D_A_fake)*0.5
                loss_D_A.backward()
                pred_real_B = D_B(real_B)
                loss_D_B_real = adv(pred_real_B, torch.ones_like(pred_real_B))
                fake_B_pool = pool_B.query([fb for fb in fake_B])
                pred_fake_B = D_B(fake_B_pool.detach())
                loss_D_B_fake = adv(pred_fake_B, torch.zeros_like(pred_fake_B))
                loss_D_B = (loss_D_B_real + loss_D_B_fake)*0.5
                loss_D_B.backward()
                d_opt.step()
                pbar.set_postfix({
                    "G": f"{loss_G.item():.3f}",
                    "D_A": f"{loss_D_A.item():.3f}",
                    "D_B": f"{loss_D_B.item():.3f}",
                })
            torch.save(G_A2B.state_dict(), os.path.join(out_dir, "photo2sketch_G.pth"))
            torch.save(G_B2A.state_dict(), os.path.join(out_dir, "sketch2photo_G.pth"))
        return os.path.join(out_dir, "photo2sketch_G.pth")

    def generate_with_controlnet(sketch_path, prompt, sd_model_id="runwayml/stable-diffusion-v1-5", controlnet_id="lllyasviel/sd-controlnet-scribble", out_dir="./outputs/controlnet_out", num_inference_steps=20, guidance_scale=7.5, device=None):
        os.makedirs(out_dir, exist_ok=True)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16 if device.type=="cuda" else torch.float32)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(sd_model_id, controlnet=controlnet, safety_checker=None, feature_extractor=None)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        if device.type == "cuda":
            pipe.enable_xformers_memory_efficient_attention()
        sketch = Image.open(sketch_path).convert("RGB")
        sketch = sketch.resize((512,512), resample=Image.BICUBIC)
        generator = torch.Generator(device=device).manual_seed(torch.randint(0, 2**31 - 1, (1,)).item())
        outputs = pipe(prompt=prompt, image=sketch, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator)
        for i, img in enumerate(outputs.images):
            save_path = os.path.join(out_dir, f"controlnet_out_{i}.png")
            img.save(save_path)
        return out_dir

    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--data_dir", required=True, type=str)
        ap.add_argument("--out_dir", default="./outputs", type=str)
        ap.add_argument("--image_size", default=256, type=int)
        ap.add_argument("--batch_size", default=4, type=int)
        ap.add_argument("--cyclegan_lr", default=2e-4, type=float)
        ap.add_argument("--cyclegan_epochs", default=100, type=int)
        ap.add_argument("--only_cyclegan", action="store_true")
        ap.add_argument("--controlnet_mode", action="store_true", help="Skip dreambooth. Use ControlNet inference after CycleGAN or standalone.")
        ap.add_argument("--sketch_for_gen", type=str, default=None)
        ap.add_argument("--gen_prompt", type=str, default=None)
        ap.add_argument("--sd_model", type=str, default="runwayml/stable-diffusion-v1-5")
        ap.add_argument("--controlnet_model", type=str, default="lllyasviel/sd-controlnet-scribble")
        ap.add_argument("--num_steps", type=int, default=20)
        ap.add_argument("--guidance", type=float, default=7.5)
        args = ap.parse_args()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dataset = PairedFacesDataset(args.data_dir, image_size=args.image_size)
        dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

        cyclegan_dir = os.path.join(args.out_dir, "cyclegan")
        g_path = train_cyclegan(dl, device, epochs=args.cyclegan_epochs, lr=args.cyclegan_lr, out_dir=cyclegan_dir)

        if args.only_cyclegan:
            print("only_cyclegan set, exiting after CycleGAN.")
            return

        if args.controlnet_mode:
            if args.sketch_for_gen is None or args.gen_prompt is None:
                print("controlnet_mode selected but no sketch_for_gen or gen_prompt provided. Exiting.")
                return
            out = generate_with_controlnet(args.sketch_for_gen, args.gen_prompt, sd_model_id=args.sd_model, controlnet_id=args.controlnet_model, out_dir=os.path.join(args.out_dir, "controlnet_gen"), num_inference_steps=args.num_steps, guidance_scale=args.guidance, device=device)
            print("ControlNet generation saved to", out)

        print("Done")

    if __name__ == "__main__":
        main()
