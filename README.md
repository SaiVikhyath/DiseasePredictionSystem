# Disease Prediction System
This app has the capability to predict the disease based on the symptoms given to it.
The following diseases can be diagnosed:<br/></br>
1. Fungal infection<br/>
2. Allergy<br/>
3. Drug Reaction<br/>
4. Diabetes<br/>
5. Bronchial Asthma<br/>
6. Migraine<br/>
7. Jaundice<br/>
8. Malaria<br/>
9. Chicken pox<br/>
10. Dengue<br/>
11. Typhoid<br/>
12. Tuberculosis<br/>
13. Common Cold<br/>
14. Pneumonia<br/>
15. Covid_19<br/>
16. Headache<br/>
</br>

---

* _**Dataset is preprocessed into a desirable format in which each attribute holds the data about a symptom and each record holds the data about a specific disease. This is performed in `preprocess()` method.**_</br></br>
* _**This data is partitioned into training and testing sets. The target attribute is labelled with the integers using `labelMap()` method.**_</br></br>
* _**This data is trained using a neural network using Keras. The model is SEQUENTIAL to be precise and the classifier is ADAM.**_</br></br>
* _**The model is a three-layer network with the input layer having 48 nodes which represents the symptoms. There are three hidden layers with 24 nodes each. The output layer has 16 nodes which represents the number of diseases.**_</br></br>
* _**RELU activation function is used for the hidden layers and SoftMax activation function is used for the output layer.**_</br></br>
* _**The value resulted by the model is unmapped to the disease using `labelUnmap()` method.**_</br></br>
* _**This model is made as a web-app using Django framework.**_</br></br>

---

## Requirements

1. Python>=3.7
    * Numpy>=1.19.2
    * Pandas>=1.1.5
    * Scikit-learn>=0.24.1
    * TensorFlow>=2.4.1
    * Keras>=2.4.3
    * pip>=21.0.1
2. Django>=3.1.7
