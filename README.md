# Induction-Motor-Fault-Diagnosis-using-Convolutional-Neural-Networks-A-Physics-Aware-Approach
Induction motors are vital to industrial machinery, making fault detection essential. This repo presents an automated MCSA-based deep learning system using MATLAB Simulink, STFT spectrograms, and CNNs to classify six motor conditions. Trained on 3,000 simulations, the model achieved 98.1% test accuracy.

This project presents an automated, physics-aware fault diagnosis system for induction motors using Motor Current Signature Analysis (MCSA) and Convolutional Neural Networks (CNN). It was developed by Muhammad Farhan Shah, Kumail Ahmad, Muhammad Ahmed, Huzyepha Javaid, and Shaheer Asghar from NUST SEECS.

Project Overview
The framework replaces manual expert analysis of vibration or current data with a deep learning pipeline that detects electromechanical motor faults under realistic operating conditions and grid fluctuations. The system models six motor operating conditions for a 7.5 kW Squirrel Cage Induction Motor (SCIM): Healthy, Broken Rotor Bar, Eccentricity Fault, Bearing Fault, Inter-Turn Stator Short, and Supply Voltage Imbalance.

System Methodology and Fault Modeling

1. Data Generation: Synthetic data was created using high-fidelity MATLAB Simulink models. The dataset contains 3,000 total simulation runs (500 per class) with 1% white noise added to reflect real-world variability.


2. Operating Parameter Randomization:

* Supply Frequency: Randomized between 49.5 Hz and 50.5 Hz to model grid frequency drift.


* Load Torque: Randomized between 10 Nm and 60 Nm to vary slip and shift frequency locations.


* Input Voltage: Randomized between 350 V and 450 V to mimic voltage sags and swells.


* Fault Severity: Parameterized across dynamic ranges from minor to severe fault states.



3. Electromechanical Fault Physical Definitions:

* Broken Rotor Bar (BRB): Modeled using Amplitude Modulation of stator current, creating sidebands around the fundamental frequency.


* Eccentricity Fault: Modeled as mixed eccentricity, modulating air-gap flux density.


* Bearing Fault: Modeled as high-frequency impulses based on the Ball Pass Frequency of the Outer Race (BPFO).


* Inter-Turn Stator Short: Modeled by injecting odd harmonics (3rd, 5th, and 7th) into current signals.


* Voltage Imbalance: Modeled by reducing Phase A voltage by 5% to 15%, creating negative sequence currents.



Signal Processing Pipeline

* Sampling and Decimation: Raw 5-second current data sampled at 10 kHz is decimated to 1 kHz to focus on the 0-400 Hz spectral range. The initial 1 second of transient startup data is trimmed.


* Spectral Analysis: Utilizes a High-Resolution Short-Time Fourier Transform (STFT) with a 0.33 Hz frequency resolution and a Chebyshev window to separate fault sidebands.


* RGB Spectrogram Mapping: Processed 3-phase currents are encoded directly into Red, Green, and Blue image channels to allow the CNN to leverage inter-phase correlation features.



CNN Model Performance

* Overall Accuracy: Achieved a test classification accuracy of 98.1% across a 70-15-15 train-validation-test split.


* Accuracy by Fault Class: Bearing Fault achieved 100%, Inter-Turn Short Circuit achieved 100%, Eccentricity Fault achieved 99.11%, Voltage Imbalance achieved 97.78%, and Broken Rotor Bar achieved 96%.


* Robustness: Vertical translation data augmentation was implemented during training to make the network robust against grid frequency drift.



How to Run the Project

1. Run the MATLAB Simulink motor models to generate 3-phase current simulation files.


2. Execute the signal processing script to decimate signals to 1 kHz, trim transients, and apply Chebyshev windowed STFT.


3. Map 3-phase STFT signals into RGB spectrogram images.


4. Train the Convolutional Neural Network on the RGB spectrogram dataset using the 70-15-15 dataset split.


5. Evaluate classification accuracy and confusion matrix on the test set.
