# Introduction

This project was developed as part of the FaceCup competition. The goal of this challenge was to design an AI-powered deep learning system capable of detecting the geographical location of an image and accurately counting the number of men and women present. Given the complexity of the task, advanced image processing models were utilized to enhance the system’s accuracy and efficiency.

![1](https://github.com/user-attachments/assets/fedafcf7-a6ee-482b-a010-1f6ee86b004e)

# Key Challenges

High Number of Classes: The dataset contained 488 unique location classes representing various tourist attractions in Iran, significantly increasing the model’s search space and making precise location identification difficult.

Limited Data per Class: Each class had a relatively small number of samples, creating challenges in deep learning training and reducing model generalization.

Integration of Person Detection and Gender Classification: While the competition’s baseline model used two separate models for these tasks, this project combined them into a single model using YOLO v11, optimizing computational efficiency and reducing inference time.


# Models Used

EfficientNet-B6: Used for geographical location detection with high generalization capability.

YOLO v11: Used for person detection and gender classification, providing higher accuracy and faster processing compared to separate models.


# Data Structure and Output Format

Input:

A collection of images depicting various tourist, historical, and urban locations in Iran.

Output:

A .csv file structured as follows:

Column 1: Image/video filename

Column 2: Number of men detected in the image

Column 3: Number of women detected in the image

Columns 4 to 491: Probability of the image belonging to each of the 488 unique locations, determined using the EfficientNet-B6 model

# Suggested Improvements

Utilizing Vision Transformers (ViTs) to enhance the model’s ability to understand complex spatial features and improve generalization.

Expanding the dataset using data augmentation techniques, such as rotation, scaling, noise addition, and color variations.

Implementing Self-Supervised Learning to reduce reliance on labeled data and leverage unlabeled samples more effectively.


# Conclusion

This project successfully integrated EfficientNet-B6 and YOLO v11 to provide an optimized and faster solution compared to the competition’s baseline model. The unified approach to person detection and gender classification resulted in reduced computational cost and improved efficiency. Furthermore, incorporating advanced techniques such as Vision Transformers and Self-Supervised Learning can further enhance the model’s accuracy and generalization capabilities in future iterations.
