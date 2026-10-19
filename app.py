import json
import logging
import os
import threading
import time
import uuid
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from flask import Flask, jsonify, redirect, render_template, request, url_for

from model import train_models


app = Flask(__name__)
logger = logging.getLogger(__name__)

rf, gb, feature_names = train_models()
symptom_data = [(col, col.replace("_", " ").title()) for col in feature_names]

ai_jobs = {}
ai_lock = threading.Lock()
AI_JOBS_MAX_AGE_SECONDS = 60 * 60
HF_CHAT_COMPLETIONS_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct-1M")

precautions = {
    "Flu": "Rest, fluids, and paracetamol if needed.",
    "Cold": "Stay warm, drink hot fluids.",
    "Migraine": "Rest in a dark, quiet room.",
    "Malaria": "Seek medical help immediately.",
    "Dengue": "Hydration is critical, consult doctor.",
    "Allergy": "Avoid triggers, take antihistamines.",
    "Viral Fever": "Rest, fluids, light diet.",
    "Fatigue Syndrome": "Proper sleep and stress control.",
    "Bronchitis": "Rest, steam inhalation, and fluids.",
    "Sinusitis": "Steam inhalation and nasal sprays.",
    "Typhoid": "Antibiotics and clean water intake.",
    "Stress": "Relaxation and proper sleep.",
    "COVID-19": "Isolation, hydration, and consult doctor.",
    "Food Poisoning": "Oral rehydration and light food.",
    "Throat Infection": "Warm salt water gargle.",
    "Chikungunya": "Rest and pain relievers.",
    "Asthma": "Use inhaler and avoid triggers.",
    "Anxiety": "Relaxation techniques and therapy.",
}


class AIServiceError(Exception):
    pass


def text_to_symptoms(text):
    text = text.lower()
    return [symptom for symptom in feature_names if symptom.replace("_", " ") in text]


def get_medical_advice(disease):
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise AIServiceError(
            "Hosted AI is not configured. Set the HF_TOKEN environment variable "
            "to a Hugging Face token with Inference Providers access."
        )

    payload = {
        "model": HF_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You provide brief, cautious health education, not medical care. "
                    "The condition label is an unverified prediction from an educational "
                    "symptom checker, not a diagnosis. Do not prescribe medicines or "
                    "dosages, and do not tell the user to start or stop treatment. "
                    "Give up to four general safety suggestions, recommend consulting "
                    "a qualified clinician, and mention seeking urgent care for severe "
                    "or rapidly worsening symptoms."
                ),
            },
            {
                "role": "user",
                "content": f"Give general health guidance for the possible condition: {disease}.",
            },
        ],
        "temperature": 0.3,
        "max_tokens": 180,
    }
    api_request = Request(
        HF_CHAT_COMPLETIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(api_request, timeout=45) as response:
            result = json.load(response)
    except HTTPError as error:
        if error.code in (401, 403):
            raise AIServiceError(
                "Hugging Face rejected the token. Check that HF_TOKEN is valid "
                "and has Inference Providers access."
            ) from error
        if error.code == 429:
            raise AIServiceError(
                "The Hugging Face free quota or rate limit was reached. Try again later."
            ) from error
        raise AIServiceError(
            f"Hugging Face could not generate advice (HTTP {error.code}). Try again later."
        ) from error
    except URLError as error:
        if isinstance(error.reason, TimeoutError):
            raise AIServiceError("The Hugging Face request timed out. Try again later.") from error
        raise AIServiceError(
            "Could not reach Hugging Face. Check the server's internet connection."
        ) from error
    except TimeoutError as error:
        raise AIServiceError("The Hugging Face request timed out. Try again later.") from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise AIServiceError("Hugging Face returned an invalid response. Try again later.") from error

    try:
        advice = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise AIServiceError("Hugging Face returned an unexpected response.") from error

    if not isinstance(advice, str) or not advice.strip():
        raise AIServiceError("Hugging Face returned empty advice. Try again later.")

    return advice.strip()


def generate_ai_async(job_id, disease):
    try:
        advice = get_medical_advice(disease)
    except AIServiceError as error:
        logger.warning("AI advice generation failed: %s", error)
        result = {"status": "error", "error": str(error)}
    except Exception:
        logger.exception("Unexpected AI advice generation failure")
        result = {
            "status": "error",
            "error": "AI generation failed unexpectedly. Check the server log and try again.",
        }
    else:
        result = {"status": "completed", "advice": advice}

    with ai_lock:
        if job_id in ai_jobs:
            ai_jobs[job_id].update(result)


def create_ai_job(disease):
    now = time.monotonic()
    with ai_lock:
        expired_jobs = [
            job_id
            for job_id, job in ai_jobs.items()
            if now - job["created_at"] > AI_JOBS_MAX_AGE_SECONDS
        ]
        for job_id in expired_jobs:
            del ai_jobs[job_id]

        job_id = uuid.uuid4().hex
        ai_jobs[job_id] = {
            "disease": disease,
            "status": "ready",
            "created_at": now,
        }
    return job_id


@app.route("/")
def home():
    return render_template(
        "index.html",
        symptom_data=symptom_data,
        results=None,
        advice=None,
        ai_job_id=None,
        disease=None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    selected = request.form.getlist("symptoms")
    text_input = request.form.get("text_input", "")

    if text_input:
        selected += text_to_symptoms(text_input)

    input_data = [1 if symptom in selected else 0 for symptom in feature_names]
    pred1 = rf.predict([input_data])[0]
    pred2 = gb.predict([input_data])[0]
    disease = max((pred1, pred2), key=(pred1, pred2).count)
    job_id = create_ai_job(disease)

    return redirect(url_for("result", request_id=job_id))


@app.route("/result")
def result():
    request_id = request.args.get("request_id", "")
    with ai_lock:
        job = ai_jobs.get(request_id)
        disease = job["disease"] if job else None

    if not disease:
        return redirect(url_for("home"))

    return render_template(
        "index.html",
        symptom_data=symptom_data,
        results=[(disease, 100)],
        advice=precautions.get(disease, "Consult a doctor."),
        ai_job_id=request_id,
        disease=disease,
    )


@app.route("/generate_ai_advice/<request_id>", methods=["POST"])
def start_ai_advice(request_id):
    with ai_lock:
        job = ai_jobs.get(request_id)
        if not job:
            return jsonify({"error": "This result has expired. Run a new symptom check."}), 404

        if job["status"] == "generating":
            return jsonify({"status": "generating"})
        if job["status"] == "completed":
            return jsonify({"status": "completed"})

        job["status"] = "generating"
        job.pop("error", None)
        disease = job["disease"]

    threading.Thread(
        target=generate_ai_async,
        args=(request_id, disease),
        daemon=True,
    ).start()
    return jsonify({"status": "generating"})


@app.route("/get_ai_advice/<request_id>")
def get_ai_advice_api(request_id):
    with ai_lock:
        job = ai_jobs.get(request_id)
        if not job:
            return jsonify({"error": "This result has expired. Run a new symptom check."}), 404

        response = {"status": job["status"]}
        if "advice" in job:
            response["advice"] = job["advice"]
        if "error" in job:
            response["error"] = job["error"]
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
