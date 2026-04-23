from flask import Flask, render_template, request
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

model = joblib.load("customer_segmentation.pkl")

history = []

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict-page')
def predict_page():
    return render_template("index.html", history=history)



@app.route('/predict', methods=['POST'])
def predict():
    try:
        income = float(request.form['income'])
        score = float(request.form['score'])

        age = request.form.get('age')
        gender = request.form.get('gender')

        prediction = model.predict([[income, score]])
        cluster = int(prediction[0])

        labels = {
            0: "Average Customers - Maintain engagement",
            1: "High Value Customers - Target premium products",
            2: "Low Income High Spend - Offer smart discounts",
            3: "High Income Low Spend - Upsell strategy needed",
            4: "Low Value Customers - Budget campaigns"
        }

        result = labels.get(cluster, "Unknown Customer Type")

        if cluster == 1:
            recommendation = "Offer premium memberships & exclusive deals"
        elif cluster == 2:
            recommendation = "Give discounts but monitor spending behavior"
        elif cluster == 3:
            recommendation = "Push luxury product ads"
        elif cluster == 0:
            recommendation = "Keep them engaged with regular offers"
        else:
            recommendation = "Focus on low-cost marketing strategies"

        history.append({
            "income": income,
            "score": score,
            "cluster": cluster,
            "result": result
        })

        generate_plot()

        return render_template(
            "index.html",
            prediction_text=result,
            recommendation=recommendation,
            age=age,
            gender=gender,
            history=history
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text="Error: " + str(e),
            history=history
        )


def generate_plot():
  
    plt.figure()

    incomes = [h['income'] for h in history]
    scores = [h['score'] for h in history]

    plt.scatter(incomes, scores)
    plt.xlabel("Income")
    plt.ylabel("Spending Score")
    plt.title("Customer Predictions")


    if not os.path.exists("static"):
        os.makedirs("static")

    plt.savefig("static/plot.png")
    plt.close()


if __name__ == "__main__":
    app.run(debug=True)