from flask import Flask, request, jsonify, render_template
import joblib, pandas as pd
import os


app = Flask(__name__)
pipe = joblib.load("model_storage/model.joblib")



CATS = {
  "job": ['admin.','blue-collar','technician','services','management','retired',
          'entrepreneur','self-employed','housemaid','unemployed','student'],
  "marital": ['married','single','divorced'],
  "education": ['basic.4y','basic.6y','basic.9y','high.school','university.degree','professional.course','illiterate'],
  "default": ['no','yes','unknown'],
  "housing": ['no','yes'],
  "loan": ['no','yes'],
  "contact": ['cellular','telephone'],
  "month": ['mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],
  "day_of_week": ['mon','tue','wed','thu','fri'],
  "poutcome": ['nonexistent','failure','success'],
}
NUMS = ["age","duration","campaign","pdays","previous","emp.var.rate","cons.price.idx","cons.conf.idx"]
FIELDS = list(CATS.keys()) + NUMS

@app.get("/health")
def health():
    return jsonify({"status":"ok"})

@app.get("/")
def index():
    return render_template("index.html", cats=CATS, nums=NUMS)

@app.post("/predict-form")
def predict_form():
    try:
        data = {k: request.form.get(k) for k in FIELDS}

        # cast numerics
        for k in NUMS:
            v = data.get(k, "")
            data[k] = float(v) if v != "" else 0.0
            if k in ["age","duration","campaign","pdays","previous"]:
                data[k] = int(data[k])

        df = pd.DataFrame([data], columns=FIELDS)
        p = float(pipe.predict_proba(df)[0][1])
        y = int(p >= 0.5)
        return render_template("result.html", pred=y, proba=p, inputs=data)
    except Exception as e:
        return f"<pre>ERROR:\n{e}</pre>", 400


@app.post("/predict")
def predict_api():
    payload = request.get_json(force=True)
    df = pd.DataFrame([payload], columns=FIELDS)
    p = float(pipe.predict_proba(df)[0][1])
    y = int(p >= 0.5)
    return jsonify({"prediction": y, "probability": p})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)

