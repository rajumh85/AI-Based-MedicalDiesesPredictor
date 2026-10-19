from flask import Flask, render_template, request, redirect, url_for, jsonify
from model import train_models
import ollama
import threading

app = Flask(__name__)

# ===== ML MODELS =====
rf, gb, feature_names = train_models()
symptom_data = [(col, col.replace("_", " ").title()) for col in feature_names]

# ===== CACHE =====
ai_cache = {}
ai_lock = threading.Lock()

# ===== PRECAUTIONS =====
precautions = {
    "Flu": "Rest, fluids, and paracetamol if needed.",
    "Cold": "Stay warm, drink hot fluids.",
    "Migraine": "Rest in a dark, quiet room.",
    "Malaria": "Seek medical help immediately.",
    "Dengue": "Hydration is critical, consult doctor.",
    "Allergy": "Avoid triggers, take antihistamines.",
    "Viral Fever": "Rest, fluids, light diet.",
    "Fatigue Syndrome": "Proper sleep and stress control."
}
# ===== ADD MORE PRECAUTIONS =====
precautions.update({
    "Bronchitis": "Rest, steam inhalation, and fluids.",
    "Sinusitis": "Steam inhalation and nasal sprays.",
    "Typhoid": "Antibiotics and clean water intake.",
    "Stress": "Relaxation and proper sleep.",
    "COVID-19": "Isolation, hydration, and consult doctor.",
    "Food Poisoning": "Oral rehydration and light food.",
    "Throat Infection": "Warm salt water gargle.",
    "Chikungunya": "Rest and pain relievers.",
    "Asthma": "Use inhaler and avoid triggers.",
    "Anxiety": "Relaxation techniques and therapy."
})

# ===== NLP =====
def text_to_symptoms(text):
    text = text.lower()
    return [s for s in feature_names if s.replace("_", " ") in text]

# ===== FAST AI FUNCTION =====
def get_medical_advice(disease):
    try:
        response = ollama.chat(
            model='phi3',  # ⚡ fast model
            messages=[{
                'role': 'user',
                'content': f"""
Give quick medical advice for {disease}.

Rules:
- Only 5 to 6 bullet points
- Each point must be 1 short line
- Include medicines and precautions
- No explanations
"""
            }],
            options={
                "temperature": 0.3,
                "num_predict": 120
            }
        )

        return response['message']['content']

    except Exception as e:
        print("OLLAMA ERROR:", e)
        return "AI advice failed."

# ===== ASYNC THREAD =====
def generate_ai_async(disease):
    with ai_lock:
        ai_cache[disease] = get_medical_advice(disease)

# ===== HOME =====
@app.route("/")
def home():
    return render_template("index.html",
                           symptom_data=symptom_data,
                           results=None,
                           advice=None,
                           ai_advice=None,
                           disease=None)

# ===== PREDICT =====
@app.route("/predict", methods=["POST"])
def predict():

    selected = request.form.getlist("symptoms")
    text_input = request.form.get("text_input")

    if text_input:
        selected += text_to_symptoms(text_input)

    input_data = [1 if s in selected else 0 for s in feature_names]

    pred1 = rf.predict([input_data])[0]
    pred2 = gb.predict([input_data])[0]

    final_prediction = max([pred1, pred2], key=[pred1, pred2].count)

    # Clear old cache
    ai_cache.clear()

    # Start AI generation in background
    threading.Thread(
        target=generate_ai_async,
        args=(final_prediction,),
        daemon=True
    ).start()

    return redirect(url_for('result', disease=final_prediction))

# ===== RESULT PAGE =====
@app.route("/result")
def result():

    disease = request.args.get("disease")
    advice = precautions.get(disease, "Consult a doctor.")

    return render_template("index.html",
                           symptom_data=symptom_data,
                           results=[(disease, 100)],
                           advice=advice,
                           ai_advice="⏳ Generating quick AI tips...",
                           disease=disease)

# ===== LIVE AI FETCH =====
@app.route("/get_ai_advice")
def get_ai_advice_api():
    disease = request.args.get("disease")
    return jsonify({"advice": ai_cache.get(disease)})

# ===== RUN =====
if __name__ == "__main__":
    app.run(debug=True)