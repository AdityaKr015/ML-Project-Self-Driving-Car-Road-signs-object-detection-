My first Machine Learning project built with my friend [Vaibhav](https://github.com/vaibhav410) for college project, where we developed a Deep Learning-based Road Sign Detection system. 


🚦 **Road Sign Detection using Deep Learning**


📌 **Project Overview**


This project focuses on detecting and classifying road signs from images using deep learning.
The goal is to assist self-driving cars in understanding road environments by recognizing traffic signals, speed limits, and other signs.


🛠️ **Features**


Real-time road sign detection.

Trained on labeled dataset of traffic signs.

Supports classification of multiple road sign categories.


**Implementation**


The implementation of the Object Detection project was carried out in a systematic manner, beginning with dataset preparation and ending with real-time detection results. The entire workflow was executed in Google Colab using GPU acceleration for efficient training and testing.


1. **Dataset Integration**

•	The dataset was created and managed using Roboflow, which allowed image uploading, annotation, and preprocessing.
•	Each image was annotated with bounding boxes marking the location of objects.
•	The final dataset was exported in YOLO-compatible format (with training, validation, and test splits).
•	The dataset was then imported into the Colab environment for model training.


2. **Environment Setup**
 
•	In Google Colab, the required dependencies were installed, including:
o	Ultralytics (for YOLOv8 framework)
o	Pandas (for data handling and analysis)
o	Roboflow API (for dataset access and integration)
•	GPU runtime was enabled to accelerate training.
•	Project directories were created for organizing datasets, model weights, and results.


3. **Model Training**
 
•	The YOLOv8 model from Ultralytics was selected due to its high speed and accuracy.
•	Pre-trained YOLOv8 weights were used and fine-tuned on the custom dataset (transfer learning).
•	Training parameters:
o	Epochs: Defined number of iterations for learning.
o	Batch size: Optimized for available GPU memory.
o	Learning rate: Tuned for stable convergence.


4. **Model Validation**
 
•	After training, the model was validated on a separate test dataset.
•	Key metrics were measured, including:
o	Precision: How many predicted objects were correct.
o	Recall: How many actual objects were detected.
o	mAP (mean Average Precision): Overall detection performance.
•	The validation process helped evaluate how well the model generalized to unseen images.


5. **Inference and Output**
 
•	The trained YOLOv8 model was used for inference on new test images.
•	Each detected object was marked with a bounding box, label, and confidence score.
•	Results showed that the model successfully detected multiple objects simultaneously and delivered accurate predictions.
•	The outputs demonstrated the system’s ability to be applied in real-world scenarios such as surveillance, product detection, and automation tasks.



<img width="975" height="731" alt="image" src="https://github.com/user-attachments/assets/c165f5eb-ad0f-4ba9-94b5-1211483cdb71" />



<img width="975" height="488" alt="image" src="https://github.com/user-attachments/assets/1d7616d6-2551-431f-9a48-12bbfc4c5758" />





**WORKFLOW FOR THIS PROJECT:**



<img width="917" height="617" alt="image" src="https://github.com/user-attachments/assets/7d0114f3-94a6-484c-b8c3-b0a615ab96a7" />




📂 **Project Structure**

├── object_detection3.ipynb   # Main Jupyter Notebook

├── dataset/                  # Training & testing images

├── models/                   # Saved models (if any)

├── requirements.txt          # Dependencies

└── README.md                 # Project documentation



⚙️ **Tech Stack**

Language: Python

Frameworks/Libraries: Ultralytics (YOLOv8 Framework),Pandas,Roboflow,ipython

Tools: Google Colab



📊 Dataset

Dataset used:

https://universe.roboflow.com/selfdriving-car-qtywx/self-driving-cars-lfjou

Contains thousands of labeled images of road signs.


🚀 **How to Run**


Clone the repository
```bash
git clone https://github.com/<your-username>/road-sign-detection.git
cd road-sign-detection
```

Install dependencies
```bash
pip install -r requirements.txt
```

Run the Jupyter Notebook
```
jupyter notebook object_detection3.ipynb
```


📈 **Results**


<img width="975" height="548" alt="image" src="https://github.com/user-attachments/assets/169c3b14-2148-49bc-8ae6-f48fa0d31129" />



<img width="975" height="550" alt="image" src="https://github.com/user-attachments/assets/ff57cc6f-a543-4dbf-a5e7-1b172bc5d7c3" />



<img width="975" height="549" alt="image" src="https://github.com/user-attachments/assets/9e7a03e8-69b4-4762-b97a-a3ba4ac9ff01" />




Model accuracy: Overall 85 to 90 % accuracy 

Correctly classifies speed limits, stop signs, and other common traffic symbols.


Visualization of detected signs included in the notebook.

👩‍💻 **Contributors**

#[Aditya](https://github.com/AdityaKr015)

#[Vaibhav](https://github.com/vaibhav410)
 

