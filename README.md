# SynapticSketch-AI

A project for sketch/photo synthesis and matching — training, inference, and a simple web UI. Contains Python training/inference scripts, a `frontend` app, datasets, and outputs.

**Contents**
- `train.py` — training script
- `infer.py`, `infer_entry.py`, `infer_clean.py` — inference utilities
- `webapp.py` — simple web UI / backend entrypoint
- `generate_test_graphs.py` — plotting/visualization utilities
- `dataset/` — source images (real + sketches)
- `outputs/` — generated images, graphs, checkpoints
- `frontend/` — web frontend (Vite + React/TypeScript)

**Requirements**
- Python 3.8+
- Node.js (for frontend)
- Recommended: create and use a virtual environment

**Quick Start**

1. Create and activate a Python virtual environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Start the backend (if `webapp.py` is a Flask/FastAPI app):

```powershell
python webapp.py
```

3. Start the frontend (optional):

```powershell
cd frontend
npm install
npm run dev
# or use pnpm/yarn depending on your setup
```

4. Run training or inference as needed (examples):

```powershell
python train.py
python infer.py --input dataset/real/pic1.jpg --output outputs/inference_result.png
python generate_test_graphs.py
```

Check the top of `train.py` / `infer.py` for CLI flags or configuration options.

**Changing the dataset**

If you want to replace or change the dataset used by training/inference, follow these steps:

1. Prepare your dataset locally. Keep the layout compatible with existing code, for example:

```
dataset/
  real/        # photos
    img001.jpg
    img002.jpg
  sketches/    # sketches that correspond to photos
    img001.png
    img002.png
```

2. Point the scripts to the new dataset:
- Option A: Edit the path variables or config at the top of `train.py` / `infer.py` to point to your new folders.
- Option B: If the script supports CLI args or environment variables, supply the new paths when running the script.

3. Avoid committing large raw images to the repository (recommended):

- Add dataset folders you don't want checked in to `.gitignore`. Example to ignore `dataset/real/`:

```text
/dataset/real/
/dataset/sketches/
```

- If images were already committed and you want to remove them from the repo while keeping them locally, run:

```powershell
git rm -r --cached dataset/real
git rm -r --cached dataset/sketches
git commit -m "Remove dataset images from repo; add to .gitignore"
git push
```

4. If large files were committed in the past and you need to remove them from history, consider using tools such as the BFG Repo-Cleaner or `git filter-repo`. These tools rewrite history — follow their documentation and back up your repo first.

5. After switching datasets, re-run training and inference to regenerate outputs and checkpoints:

```powershell
python train.py
python generate_test_graphs.py
python infer.py --input path/to/test.jpg --output outputs/new_inference.png
```

**Model checkpoints and outputs**

- The project `.gitignore` excludes virtualenv folders, `node_modules/`, and generated model files under `outputs/` (for example `*.pth`). Model binaries should not be checked into the repository.

**If you want this repo to stop tracking dataset files already committed**

- Add dataset paths to `.gitignore`.
- Remove cached copies from git index (see `git rm --cached` above) and commit.

**Contributing**

- Use feature branches and open PRs for changes.
- Describe reproducible steps for training/inference and include sample commands.


---
