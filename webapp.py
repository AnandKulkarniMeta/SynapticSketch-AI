from flask import Flask, request, render_template, send_from_directory, redirect, url_for
from flask import jsonify
from flask_cors import CORS
import os
import time
import shutil
from werkzeug.utils import secure_filename
import json

import infer_clean

app = Flask(__name__, static_folder="outputs", template_folder="templates")

# Allow cross-origin requests from local frontend during development
CORS(app)

OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    result_url = None
    message = None

    if request.method == "POST":
        prompt = request.form.get("prompt", "").strip()

        if not prompt:
            message = "Prompt is required."
            return render_template("index.html", result_url=result_url, message=message)

        # choose a timestamped output filename
        basename = f"web_result_{int(time.time())}.png"
        out_path = os.path.join(OUTPUT_DIR, basename)

        try:
            # 1) Try to find a matching real image for the prompt
            try:
                match = infer_clean.find_prompt_match(prompt, "./dataset", mode="exact", fuzzy_threshold=0.85, target="real")
            except TypeError:
                # fallback if signature differs
                match = infer_clean.find_prompt_match(prompt, "./dataset", mode="exact", fuzzy_threshold=0.85)

            if match is not None:
                # Found a matched real image — convert it with CycleGAN to a sketch
                matched_img, score, matched_txt = match
                cyc_path = os.path.join("./outputs/cyclegan", "photo2sketch_G.pth")
                if not os.path.exists(cyc_path):
                    raise FileNotFoundError(f"CycleGAN model not found at {cyc_path}")
                G = infer_clean.load_cyclegan_generator(cyc_path)
                infer_clean.infer_cyclegan(G, matched_img, out_path)
                message = f"Matched dataset real image and converted to sketch (score={score:.3f})."
            else:
                # No match: generate a new photo with Stable Diffusion and convert to sketch with CycleGAN
                tmp_sd = out_path + ".sdtmp.png"
                infer_clean.infer_stable_diffusion("runwayml/stable-diffusion-v1-5", prompt, tmp_sd, num_inference_steps=40, guidance_scale=7.5)
                cyc_path = os.path.join("./outputs/cyclegan", "photo2sketch_G.pth")
                if os.path.exists(cyc_path):
                    G = infer_clean.load_cyclegan_generator(cyc_path)
                    infer_clean.infer_cyclegan(G, tmp_sd, out_path)
                    try:
                        os.remove(tmp_sd)
                    except Exception:
                        pass
                    message = "Generated new photo with SD and converted to sketch via CycleGAN."
                else:
                    # CycleGAN missing: keep SD output
                    os.rename(tmp_sd, out_path)
                    message = "Generated new photo with SD (CycleGAN not found; saved photo)."

            # Use Post/Redirect/Get to avoid the browser re-sending the form and to clear the textarea
            return redirect(url_for("index", result=basename, message=message))
        except Exception as e:
            message = f"Failed to produce image: {e}"
            return redirect(url_for("index", message=message))

    # GET: read result/message from query params (PRG)
    arg_result = request.args.get("result")
    if arg_result:
        result_url = url_for('static', filename=arg_result)
    message = request.args.get("message")

    return render_template("index.html", result_url=result_url, message=message)

@app.route('/images/<path:filename>')
def serve_image(filename):
    # Serve images from the project's static/images folder
    img_dir = os.path.join(os.getcwd(), 'static', 'images')
    return send_from_directory(img_dir, filename)

@app.route('/api/db', methods=['GET'])
def api_db():
    # Return the criminal_data.json content with adjusted local image URLs
    data_path = os.path.join(os.getcwd(), 'criminal_data.json')
    if not os.path.exists(data_path):
        return jsonify([])
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return jsonify([])

    # Adjust profile_image to point to our /images/<filename> endpoint when possible
    for entry in data:
        img_url = entry.get('profile_image')
        if img_url:
            basename = os.path.basename(img_url)
            # set a local image url relative to this server
            entry['_local_image'] = url_for('serve_image', filename=basename, _external=False)
        else:
            entry['_local_image'] = None

    return jsonify(data)


@app.route('/api/generate', methods=['POST'])
def api_generate():
    """Generate or convert an image for a prompt and return JSON with result and match info.

    This reuses the same logic as the HTML form handler but returns structured JSON so
    frontends don't need to parse HTML.
    """
    prompt = request.form.get('prompt', '').strip()
    if not prompt:
        return jsonify({'error': 'prompt required'}), 400

    # timestamped filename
    import time
    basename = f"web_result_{int(time.time())}.png"
    out_path = os.path.join(OUTPUT_DIR, basename)

    try:
        # 1) Try to find a matching real image for the prompt
        try:
            match = infer_clean.find_prompt_match(prompt, "./dataset", mode="exact", fuzzy_threshold=0.85, target="real")
        except TypeError:
            match = infer_clean.find_prompt_match(prompt, "./dataset", mode="exact", fuzzy_threshold=0.85)

        message = None
        match_info = None

        if match is not None:
            matched_img, score, matched_txt = match
            cyc_path = os.path.join("./outputs/cyclegan", "photo2sketch_G.pth")
            if not os.path.exists(cyc_path):
                # If CycleGAN missing, fall back to copying matched sketch if available
                # copy matched image to outputs
                try:
                    shutil.copyfile(matched_img, out_path)
                    message = f"Matched dataset real image found; CycleGAN missing so copied matched image (score={score:.3f})."
                except Exception:
                    raise FileNotFoundError(f"CycleGAN model not found at {cyc_path} and failed to copy matched image")
            else:
                G = infer_clean.load_cyclegan_generator(cyc_path)
                infer_clean.infer_cyclegan(G, matched_img, out_path)
                message = f"Matched dataset real image and converted to sketch (score={score:.3f})."

            # prepare match_info
            img_filename = os.path.basename(matched_img) if isinstance(matched_img, str) else None
            matched_image_url = url_for('serve_image', filename=img_filename, _external=False) if img_filename else None
            try:
                if isinstance(matched_txt, str) and os.path.exists(matched_txt):
                    with open(matched_txt, 'r', encoding='utf-8') as f:
                        matched_txt_content = f.read()
                else:
                    matched_txt_content = str(matched_txt)
            except Exception:
                matched_txt_content = None

            match_info = {
                'image_filename': img_filename,
                'matched_image_url': matched_image_url,
                'score': float(score) if score is not None else None,
                'matched_txt_content': matched_txt_content,
            }

        else:
            # No match: generate with SD then convert via CycleGAN if available
            tmp_sd_out = out_path + ".sdtmp.png"
            infer_clean.infer_stable_diffusion("runwayml/stable-diffusion-v1-5", prompt, tmp_sd_out, num_inference_steps=40, guidance_scale=7.5)

            cyc_path = os.path.join("./outputs/cyclegan", "photo2sketch_G.pth")
            if os.path.exists(cyc_path):
                try:
                    G = infer_clean.load_cyclegan_generator(cyc_path)
                    infer_clean.infer_cyclegan(G, tmp_sd_out, out_path)
                    try:
                        os.remove(tmp_sd_out)
                    except Exception:
                        pass
                    message = "Generated new photo with SD and converted to sketch via CycleGAN."
                except Exception as e:
                    # leave SD output
                    print(f"CycleGAN conversion failed: {e}")
                    shutil.move(tmp_sd_out, out_path)
                    message = "Generated new photo with SD (CycleGAN conversion failed; saved SD output)."
            else:
                os.rename(tmp_sd_out, out_path)
                message = "Generated new photo with SD (CycleGAN not found; saved photo)."

        # Build a URL for the generated file (served from Flask static folder configured to `outputs`)
        result_url = url_for('static', filename=basename, _external=False)

        # If the match_info is None, also attempt a dataset match against the created image filename
        # (this mirrors the user's requested behavior: include generated result among images and try to find info)
        if match_info is None:
            try:
                # try matching against dataset sketches/reals by filename
                # This is best-effort: if the dataset contains a prompt/text file for this basename, it will be picked up
                m2 = infer_clean.find_prompt_match(prompt, "./dataset", mode="fuzzy", fuzzy_threshold=0.6, target="real")
            except Exception:
                m2 = None
            if m2 is not None:
                mm_img, mm_score, mm_txt = m2
                mm_filename = os.path.basename(mm_img) if isinstance(mm_img, str) else None
                mm_url = url_for('serve_image', filename=mm_filename, _external=False) if mm_filename else None
                try:
                    if isinstance(mm_txt, str) and os.path.exists(mm_txt):
                        with open(mm_txt, 'r', encoding='utf-8') as f:
                            mm_txt_content = f.read()
                    else:
                        mm_txt_content = str(mm_txt)
                except Exception:
                    mm_txt_content = None
                match_info = {
                    'image_filename': mm_filename,
                    'matched_image_url': mm_url,
                    'score': float(mm_score) if mm_score is not None else None,
                    'matched_txt_content': mm_txt_content,
                }

        # Log for debugging
        print(f"/api/generate prompt={prompt[:80]!r} result={basename} match={match_info}")

        return jsonify({'result_filename': basename, 'result_url': result_url, 'message': message, 'match': match_info})

    except Exception as e:
        print(f"/api/generate failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/match', methods=['POST'])
def api_match():
    prompt = request.form.get('prompt', '').strip()
    if not prompt:
        return jsonify({'match': None})

    try:
        try:
            match = infer_clean.find_prompt_match(prompt, "./dataset", mode="exact", fuzzy_threshold=0.6, target="real")
        except TypeError:
            match = infer_clean.find_prompt_match(prompt, "./dataset", mode="exact", fuzzy_threshold=0.6)
    except Exception:
        match = None

    # Log the match lookup for debugging
    print(f"/api/match lookup prompt={prompt[:80]!r} match={match}")

    if not match:
        return jsonify({'match': None})

    matched_img, score, matched_txt = match
    # matched_img may be a path to a dataset image; try to return filename and local URL
    img_filename = None
    matched_image_url = None
    if isinstance(matched_img, str):
        img_filename = os.path.basename(matched_img)
        matched_image_url = url_for('serve_image', filename=img_filename, _external=False)

    # If matched_txt is a path to a prompt file, try to read its contents
    matched_txt_content = None
    try:
        if isinstance(matched_txt, str) and os.path.exists(matched_txt):
            with open(matched_txt, 'r', encoding='utf-8') as f:
                matched_txt_content = f.read()
        else:
            # otherwise include it verbatim
            matched_txt_content = str(matched_txt)
    except Exception:
        matched_txt_content = None

    return jsonify({'match': {
        'image_filename': img_filename,
        'matched_image_url': matched_image_url,
        'score': float(score) if score is not None else None,
        'matched_txt_content': matched_txt_content
    }})


if __name__ == "__main__":
    # Enable debug for local use
    app.run(host="0.0.0.0", port=7860, debug=True)
