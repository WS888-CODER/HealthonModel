from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# تحميل النماذج المحفوظة
classifier = joblib.load('classifier_model.pkl')
regressor = joblib.load('regressor_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')

# الميزات المطلوبة
input_features = ['temperature', 'humidity', 'city', 'air_quality', 'health_status', 'allergies', 'age_group']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # التحقق من أن كل المداخل موجودة
    if not all(feature in data for feature in input_features):
        return jsonify({'error': 'Missing input fields'}), 400

    # تجهيز البيانات
    row = []
    for feature in input_features:
        val = data[feature]
        if feature in label_encoders:
            val = label_encoders[feature].transform([val])[0]
        row.append(val)

    # تحويل وقياس الميزات
    X_scaled = scaler.transform([row])

    # التنبؤ بالتصنيفات
    y_class_pred = classifier.predict(X_scaled)[0]

    # التنبؤ بالاحتمالية
    y_reg_pred = regressor.predict(X_scaled)[0][0]  # استخرجنا القيمة من [[x]] ← x فقط

    # إرسال النتائج
    return jsonify({
        'predicted_disease': y_class_pred[0],
        'predicted_risk_level': y_class_pred[1],
        'recommended_vaccine': y_class_pred[2],
        'preventive_measures': y_class_pred[3],
        'disease_probability_next_week': round(float(y_reg_pred), 3)
    })

if __name__ == '__main__':
    app.run(debug=True)
