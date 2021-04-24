from django.shortcuts import render
from django.contrib import messages
import numpy as np
import pandas as pd
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.model_selection import *
from keras.layers import *
from keras.models import *


# Create your views here.
def home(request):
    inp = []
    if request.method == "POST":
        if request.POST['abdominal_pain'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['acidity'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['back_pain'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['belly_pain'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['blurred_and_distorted_vision'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['breathlessness'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['chest_pain'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['chills'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['congestion'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['constipation'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['continuous_sneezing'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['cough'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['dark_urine'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['depression'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['diarrhoea'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['excessive_hunger'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['headache'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['high_fever'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['increased_appetite'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['indigestion'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['irritability'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['joint_pain'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['lethargy'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['loss_of_appetite'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['loss_of_smell'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['loss_of_taste'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['mild_fever'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['muscle_pain'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['pain_behind_the_eyes'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['red_spots_over_the_body'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['redness_of_eyes'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['restlessness'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['runny_nose'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['shivering'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['sinus_pressure'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['skin_rash'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['sore_throat'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['stiff_neck'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['stomach_pain'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['sweating'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['throat_irritation'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['tiredness'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['vomiting'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['watering_from_eyes'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['weight_loss'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['yellowing_of_eyes'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['yellowish_skin'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        if request.POST['itching'] == 'YES':
            inp.append(1)
        else:
            inp.append(0)
        #Remove comments to train
        #file = r'dataset.csv'
        #data  = pd.read_csv(file)
        #data = preprocess(data)
        #data.to_csv('DiseasesDataset.csv')
        #NeuralNetwork()
        model = load_model('model.h5')
        if all(v == 0 for v in inp):
            messages.error(request, 'No Illness diagnosed')
            messages.error(request, 'If you feel concerned, please consult a doctor')
        else:
            messages.info(request, 'Diagnosed : ' + str(labelUnmap(np.argmax(model.predict([[inp]])))))
            messages.info(request, '***This model only detects certain common disease.***')
            messages.info(request, '**This model is recommmended to be used only for reference and as early diagnosis tool**')
            messages.error(request, '**Please do get tested if you are diagnosed with any disease here.**')
        return render(request, 'home.html')
    else:
        return render(request, 'home.html')


def preprocess(data):
    data = data.drop_duplicates()   #Drop duplicate values
    data = data.drop(data[data.Disease == 'Dimorphic hemmorhoids(piles)'].index)
    data = data.drop(data[data.Disease == 'Hypertension '].index)
    data = data.drop(data[data.Disease == 'hepatitis A'].index)
    data = data.drop(data[data.Disease == 'Arthritis'].index)
    data = data.drop(data[data.Disease == '(vertigo) Paroymsal  Positional Vertigo'].index)
    data = data.drop(data[data.Disease == 'Psoriasis'].index)
    data = data.drop(data[data.Disease == 'GERD'].index)
    data = data.drop(data[data.Disease == 'Chronic cholestasis'].index)
    data = data.drop(data[data.Disease == 'Peptic ulcer diseae'].index)
    data = data.drop(data[data.Disease == 'Impetigo'].index)
    data = data.drop(data[data.Disease == 'AIDS'].index)
    data = data.drop(data[data.Disease == 'Gastroenteritis'].index)
    data = data.drop(data[data.Disease == 'Cervical spondylosis'].index)
    data = data.drop(data[data.Disease == 'Paralysis (brain hemorrhage)'].index)
    data = data.drop(data[data.Disease == 'Hepatitis A'].index)
    data = data.drop(data[data.Disease == 'Hepatitis B'].index)
    data = data.drop(data[data.Disease == 'Hepatitis C'].index)
    data = data.drop(data[data.Disease == 'Hepatitis D'].index)
    data = data.drop(data[data.Disease == 'Hepatitis E'].index)
    data = data.drop(data[data.Disease == 'Alcoholic hepatitis'].index)
    data = data.drop(data[data.Disease == 'Heart attack'].index)
    data = data.drop(data[data.Disease == 'Varicose veins'].index)
    data = data.drop(data[data.Disease == 'Hypothyroidism'].index)
    data = data.drop(data[data.Disease == 'Hyperthyroidism'].index)
    data = data.drop(data[data.Disease == 'Hypoglycemia'].index)
    data = data.drop(data[data.Disease == 'Osteoarthristis'].index)
    data = data.drop(data[data.Disease == '(vertigo) Paroymsal Positional Vertigo'].index)
    data = data.drop(data[data.Disease == 'Urinary tract infection'].index)
    data = data.drop(data[data.Disease == 'Acne'].index)
    cols = [i for i in data.iloc[:, 1:].columns]    #Rearranging the dataset
    tmp = pd.melt(data.reset_index(), id_vars=['index'], value_vars=cols)
    tmp['add1'] = 1
    diseases = pd.pivot_table(tmp, values='add1', index='index', columns='value')
    diseases.insert(0, 'label', data['Disease'])
    diseases = diseases.fillna(0)
    print(diseases)
    diseases = diseases.drop([' blood_in_sputum'], axis=1)
    diseases = diseases.drop([' burning_micturition'], axis=1)
    diseases = diseases.drop([' family_history'], axis=1)
    diseases = diseases.drop([' fatigue'], axis=1)
    diseases = diseases.drop([' irregular_sugar_level'], axis=1)
    diseases = diseases.drop([' malaise'], axis=1)
    diseases = diseases.drop([' mucoid_sputum'], axis=1)
    diseases = diseases.drop([' nausea'], axis=1)
    diseases = diseases.drop([' obesity'], axis=1)
    diseases = diseases.drop([' phlegm'], axis=1)
    diseases = diseases.drop([' polyuria'], axis=1)
    diseases = diseases.drop([' rusty_sputum'], axis=1)
    diseases = diseases.drop([' spotting_ urination'], axis=1)
    diseases = diseases.drop([' swelled_lymph_nodes'], axis=1)
    diseases = diseases.drop([' toxic_look_(typhos)'], axis=1)
    diseases = diseases.drop([' dischromic _patches'], axis=1)
    diseases = diseases.drop([' fast_heart_rate'], axis=1)
    diseases = diseases.drop([' visual_disturbances'], axis=1)
    diseases = diseases.drop([' nodal_skin_eruptions'], axis=1)
    diseases = diseases.drop_duplicates()
    diseases = diseases.append(diseases)
    diseases = diseases.append(diseases)
    return diseases

def NeuralNetwork():
    data = pd.read_csv('DiseasesDataset.csv')
    X = data[[' abdominal_pain', ' acidity', ' back_pain', ' belly_pain', ' blurred_and_distorted_vision', ' breathlessness',
            ' chest_pain', ' chills', ' congestion', ' constipation', ' continuous_sneezing', ' cough', ' dark_urine', ' depression',
            ' diarrhoea', ' excessive_hunger', ' headache', ' high_fever', ' increased_appetite', ' indigestion', ' irritability',
            ' joint_pain', ' lethargy', ' loss_of_appetite', ' loss_of_smell', ' loss_of_taste', ' mild_fever', ' muscle_pain',
            ' pain_behind_the_eyes', ' red_spots_over_body', ' redness_of_eyes', ' restlessness',
            ' runny_nose', ' shivering', ' sinus_pressure', ' skin_rash', ' sore_throat', ' stiff_neck', ' stomach_pain',
            ' sweating', ' throat_irritation', ' tiredness', ' vomiting', ' watering_from_eyes', ' weight_loss',
            ' yellowing_of_eyes', ' yellowish_skin', 'itching']]


    Y = data[['label']]
    Y = labelMap(Y)
    Y = np.asarray(Y).astype('float32').reshape(-1, 1)
    ohe = OneHotEncoder()
    Y = ohe.fit_transform(Y).toarray()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)
    model = Sequential()
    model.add(Dense(24, input_dim = 48))
    model.add(Dense(24, activation = 'relu'))
    model.add(Dense(24, activation = 'relu'))
    model.add(Dense(16, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    history = model.fit(X_train, Y_train, epochs = 50, batch_size = 9)
    history = model.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = 20)
    Y_pred = model.predict(X_test)
    pred = list()
    for i in range(len(Y_pred)):
        pred.append(np.argmax(Y_pred[i]))
    test = list()
    for i in range(len(Y_test)):
        test.append(np.argmax(Y_test[i]))
    a = accuracy_score(pred,test)
    print("Accuracy of the model: ", a*100)
    model.save('model.h5')

def labelMap(d):
    labels = ["Fungal infection", "Allergy", "Drug Reaction", "Diabetes ", "Bronchial Asthma",
              "Migraine", "Jaundice", "Malaria", "Chicken pox", "Dengue", "Typhoid",
              "Tuberculosis", "Common Cold", "Pneumonia", "Covid_19", "Headache"]
    Y = list()
    for i in d.iteritems():
        for j in i[1].iteritems():
            Y.append(labels.index(j[1]))
    Y = pd.Series(Y)
    return Y

def labelUnmap(value):
    labels = ["Fungal infection", "Allergy", "Drug Reaction", "Diabetes ", "Bronchial Asthma",
              "Migraine", "Jaundice", "Malaria", "Chicken pox", "Dengue", "Typhoid",
              "Tuberculosis", "Common Cold", "Pneumonia", "Covid_19", "Headache"]
    return labels[value]