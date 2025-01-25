from django.shortcuts import render
from .forms import DiabetesForm
from .models import predict_diabetes
import json

def diabetes_prediction_view(request):
    result = None
    detailed_report = {}
    if request.method == 'POST':
        form = DiabetesForm(request.POST)
        if form.is_valid():
            # Collect input data as a list
            glucose = form.cleaned_data['glucose']
            # bmi = form.cleaned_data['bmi']
            weight = form.cleaned_data['weight']
            height = form.cleaned_data['height']
            feet = int(height)  
            inches = (height - feet) * 10  
            height_in_meters = (feet * 0.3048) + (inches * 0.0254)
            bmi = weight / (height_in_meters ** 2)
            bmi = float(f"{bmi:.2f}")
            age = form.cleaned_data['age']
            insulin = form.cleaned_data['insulin']
            blood_pressure = form.cleaned_data['blood_pressure']

            # Make prediction
            result = predict_diabetes([glucose, bmi, age, insulin, blood_pressure])
            
            # Evaluate individual parameters
            detailed_report = {
                'glucose_level': {'value': glucose, 'evaluation': 'High' if glucose > 125 else 'Borderline' if glucose >= 100 else 'Normal'},
                'bmi': {'value': bmi, 'evaluation': 'Obese' if bmi >= 30 else 'Overweight' if bmi >= 25 else 'Normal'},
                'age': {'value': age, 'evaluation': '-'},
                'insulin_level': {'value': insulin, 'evaluation': 'Elevated' if insulin > 20 else 'Normal'},
                'blood_pressure': {'value': blood_pressure, 'evaluation': 'High' if blood_pressure >= 140 else 'Elevated' if blood_pressure >= 120 else 'Normal'},
                'overall_risk': result,  # "Diabetic" or "Non-Diabetic"
            }
            # summary = generate_text_gpt_neo(json.dumps(detailed_report))
            # detailed_report['summary']=summary
    else:
        form = DiabetesForm()

    return render(request, 'prediction.html', {'form': form, 'result': result, 'detailed_report': detailed_report})

def generate_text_gpt_neo(prompt, max_length=100):
    from transformers import GPTNeoForCausalLM, GPTJForCausalLM, AutoTokenizer
    # import torch
    gpt_neo_model_name = "EleutherAI/gpt-neo-2.7B" 
    gpt_neo_model = GPTNeoForCausalLM.from_pretrained(gpt_neo_model_name)
    gpt_neo_tokenizer = AutoTokenizer.from_pretrained(gpt_neo_model_name)
    inputs = gpt_neo_tokenizer(prompt, return_tensors="pt")
    outputs = gpt_neo_model.generate(inputs.input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, top_p=0.92, temperature=0.7)
    return gpt_neo_tokenizer.decode(outputs[0], skip_special_tokens=True)