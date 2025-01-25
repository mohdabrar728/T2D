from django import forms

class DiabetesForm(forms.Form):
    glucose = forms.FloatField(label='Glucose Level')
    # bmi = forms.FloatField(label='BMI')
    weight = forms.FloatField(label='Weight')
    height = forms.FloatField(label='Height')
    age = forms.IntegerField(label='Age')
    insulin = forms.FloatField(label='Insulin Level')
    blood_pressure = forms.FloatField(label='Blood Pressure')
