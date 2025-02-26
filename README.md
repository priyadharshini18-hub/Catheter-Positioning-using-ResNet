# Catheter-Positioning-using-ResNet
Multi-Label Classification of Catheter Positions in Chest X-Rays using ResNet

## Introduction
In hospital settings, patients frequently require the insertion of tubes and catheters in their body as part of the medical treatment. These devices, such as endotracheal tubes, central venous catheters, and nasogastric tubes, play a critical role in supporting essential functions like ventilation, medication delivery, and nutritional support. However, the improper placement of these tubes can result in severe complications, ranging from minor discomfort to life-threatening conditions such as pneumothorax or vascular injury. Ensuring the correct placement of these devices is, therefore, a critical task that relies heavily on the manual interpretation of chest X-rays by radiologists.

## Background
Despite being a cornerstone of patient care, the manual process of reviewing chest X-rays to verify tube placement is inherently time-consuming and subject to human error. This challenge is particularly acute in high-stress environments, such as emergency rooms or during periods of high patient loads, where the risk of oversight increases. Advances in deep learning have demonstrated immense potential in medical imaging, offering solutions to automate repetitive and error-prone tasks. By developing a system to classify catheter and tube positions as "Normal," "Borderline," or "Abnormal," this project aims to enhance the accuracy and efficiency of tube placement verification. Such a system could serve as an invaluable tool to assist radiologists, enabling faster diagnosis and timely interventions, ultimately improving patient outcomes and alleviating the burden on healthcare providers.
Problem Statement
To develop a multi-label classification model for identifying and labeling positions of catheters and tubes (ETT, NGT, CVC, Swan Ganz Catheter) in Chest X-Rays as Normal, Borderline, or Abnormal.
Terminologies
● Chest Radiograph (CXR) : A medical imaging technique that uses X-rays to create images of the chest,
including the lungs, heart, blood vessels, airways, and bones.
● ETT (Endotracheal Tube) : A flexible plastic tube inserted through the mouth into the trachea to keep the airway
open.
● NGT (Nasogastric Tube) : A special tube that carries food and medicine to the stomach through the nose.
● CVC (Central Venous Catheter) : A catheter placed into a large vein in the neck, chest, or groin to give
medication or fluids.
● Swan Ganz Catheter : A specialized catheter used to measure pressures in the heart and lungs.
   1
 Dataset
The project will utilize the comprehensive RANZCR-CLIP dataset provided by CXR the Royal Australian and New Zealand College of Radiologists on Kaggle. This dataset comprises 40,000 labeled chest X-ray images (.jpg format) generated with 11 binary classification targets covering different tube types and positions. Each image is labeled for the positioning of endotracheal tubes (ETT), nasogastric tubes (NGT), central venous catheters (CVC), and the presence of Swan Ganz catheters. The dataset includes detailed annotations and patient IDs, providing a robust foundation for developing and validating our machine learning solution.
Methodology
● Data Preprocessing : The dataset comprises chest X-ray images with multiple labels, representing different
catheter and tube positions. We performed data preprocessing to enhance the model’s performance and robustness:
○ Data Augmentation : Techniques such as rotation, flipping, and scaling were used to enhance training data
diversity and improve model generalization.
○ Cross-Validation : Implemented Group K-Fold with 4 splits, ensuring images from the same patient remain in
either training or validation sets to prevent data leakage.
○ Masking : Focused on relevant regions (lungs and chest cavity) to reduce noise and enhance model attention
on critical areas for catheter positioning using Anatomical and Intensity Based Masking.
● Model Architecture: ○ ResNeXt50 :
ResNet (Residual Networks) is a deep convolutional neural network architecture known for its use of residual connections, which help gradients flow through deep networks, enabling efficient training of very deep models. ResNeXt builds on ResNet by introducing the concept of cardinality, which refers to the number of parallel transformations within a layer, improving performance without significantly increasing model complexity.
Figure 1. Overview of ResNext50 architecture along with data preprocessing
 2

 ●
For this project, we are using ResNeXt50_32x4d, a specific variant of ResNeXt. This model has 50 layers and uses a cardinality of 32 with a width factor of 4, meaning it applies 32 parallel transformations per layer, each with a width of 4, allowing it to learn more complex features while maintaining computational efficiency. The model also consists of a custom head - a Dropout layer (0.3) for regularization, a linear layer mapping to 11 output classes and No activation function in the final layer (used with BCEWithLogitsLoss).
Training approach:
○ Transfer Learning : The ResNeXt50 model, initialized with pre-trained ImageNet weights, was fine-tuned
on the chest X-ray dataset to adapt to catheter and tube positioning, reducing training time and enhancing
performance.
○ Optimization Parameters : A batch size of 4 was chosen for memory efficiency, and the base learning rate
was set to 1e-4. Regularization was applied using weight decay at 1e-6. To prevent exploding gradients,
gradient clipping was set at a threshold of 1000.
○ Learning Rate Schedule : A CosineAnnealingLR scheduler was employed to adjust the learning rate
dynamically during training. The minimum learning rate was set to 1e-6, and the T_max value was set to 6 epochs for the cosine cycle, allowing the learning rate to gradually decrease, which helps avoid overfitting and promotes convergence.
○ Early Stopping and Model Saving : Early stopping with a patience of 3 epochs was used to halt training when validation loss stopped improving. The best model was saved based on the highest validation AUC score, ensuring optimal generalization.
We experimented with two approaches: with masking and without masking. The model's performance was evaluated using AUC (Area under Curve) scores, training and validation accuracy, and training and validation loss.
Results & Discussion
We evaluated our model's performance using two distinct approaches: with and without anatomical masking. Both implementations leveraged the ResNext50_32x4d architecture, but differed in their application of masking techniques.
● Without masking (Figure 2), Initial Loss: Started at ~0.33 (training) and ~0.24 (validation) Final Convergence: Both training and validation loss converged to ~0.19.
AUC Score: Peaked at ~0.91
Accuracy: Reached maximum of 92% for both training and validation.
Figure 2. ResNext50 Model without Masking
 3

● With Masking (Figure 3), Initial Loss: Started at ~0.27 (training) and ~0.24 (validation) Final Convergence: Achieved lower final loss of ~0.17
AUC Score: Improved performance with peak of ~0.93
Accuracy: Achieved superior accuracy of 93%
Figure 3. ResNext50 Model with Masking
Conclusion & Future Scope
In this project, we employed the ResNeXt50_32x4d model, fine-tuned on a chest X-ray dataset, to effectively classify catheter and tube positions using a combination of data augmentation and masking techniques. The results suggest that this approach, leveraging transfer learning significantly improved performance while reducing training time. Future work could focus on optimizing the model further through hyperparameter tuning, longer training periods, and testing different learning rate schedules. Additionally, expanding the model to handle other types of medical devices or applying it to other areas of medical imaging could enhance its practical applications in clinical settings, supporting more efficient and accurate diagnostic workflows.
