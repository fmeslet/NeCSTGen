======================================================================================
NeCSTGen: An approach for realistic network traffic generation using Deep Learning
======================================================================================

Summary
------------

NeCSTGen (Network Clustering Sequential Traffic Generation) is the Deep Learning architecture used 
to classify generate network traffic. The model reproduces the original behaviour at the packet, 
flow and aggregate levels. This work is published in IEEE GLOBECOM 2022 : 



Training
------------

The generation files are present in the ``training`` folder for each associated level. 
The levels are as follows:  

* Packet level  
* Flow level  
* Agregate level  



Packet level
^^^^^^^^^^^^

We sample a cluster identified in the latent space thanks to the GMM (Gaussian Mixture Model) 
and we use the VAE (Variational AutoEncoder) decoder to reconstruct the parameters of the packet. 
The following scripts are to be used:  

* ``script_vae_*_training.py``: allows the learning of the VAE (Variational AutoEncoder). There is a different script for each data set.  
* ``script_gmm_training.py``: shows how to train the GMM (Gaussian Mixture Model).  



Flow level
^^^^^^^^^^^^

* ``script_lstm_scapy_flow_connect_training``: allows the learning of the LSTM (Long Short-Term Memory) model for the generation of a flow in connected mode. For example, applications using TCP (Transport Control Protocol), the start and end structure of the flow will be taken into account.  
* ``script_lstm_scapy_flow_no_connect_training``: allows the learning of the LSTM (Long Short-Term Memory) model for the generation of a flow in unconnected mode. For example, a communication between two devices in UDP (User Datagram Protocol) mode.  
* ``script_lstm_scapy_no_flow_*_training``: allows the learning of the LSTM (Long Short-Term Memory) model for the generation of a set of packets without flow information. The "LoRaWAN" version is specific to LoRaWAN data.  



Agregate level
^^^^^^^^^^^^^^

* ``script_flows_generation_scapy.py``: shows how to train the GMM (Gaussian Mixure Model) 
allowing the generation of the characteristics of a flow.  



Generation
------------

The generation files are present in the ``inference`` folder for each associated level. 
The levels are the following: 

* Packet level   
* Flow level  
* Agregate level   



Packet level
^^^^^^^^^^^^

We sample a cluster identified in the latent space thanks to the GMM (Gaussian Mixture Model) 
and we use the VAE (Variational AutoEncoder) model decoder to reconstruct the parameters of 
the packet. The following scripts are to be used:  

* ``script_packet_generation.py``: shows how to generate a packet with the GMM (Gaussian 
Mixture Model) and the VAE (Variational AutoEncoder).  



Flow level
^^^^^^^^^^^^

Each file contains a function ``gen_pcap`` within this function the pieces of code allowing 
the generation of the ``.pcap`` file are commented. The files are the following:  

* ``script_packet_generation_scapy_flow_connect_*.py``: allows the generation of flows in connected mode. For example, applications using TCP (Transport Control Protocol), the start and end structure of the flow will be taken into account. The "Google Home" version is specific to Google Home data.  
* ``script_packet_generation_scapy_flow_no_connect_*.py``: allows the generation of a flow in unconnected mode. For example, a communication between two devices in UDP (User Datagram Protocol) mode. The "Google Home" version is specific to Google Home data.  
* ``script_packet_generation_scapy_no_flow_*py``: allows the generation of a set of packets without flow information. The "LoRaWAN" version is specific to LoRaWAN data.  

The generation of a set of flows can be done by following the dynamics present in the original data (a dynamic present over a particular period) or by using an aggregate of generated flows.  


Agregate level
^^^^^^^^^^^^^^

* ``script_flows_generation_scapy.py``: allows the generation of a set of flows but not the associated packets. A flow level generation will have to be used to be able to generate a set of packets.  


Notebooks
------------

The notebook in the directory is the one used to generate the graphs to analyze the traffic 
generated.  



Processing
------------


The folder includes all the files that have allowed to process the data and to transform them to 
make the modeling. The files present as well as their order of use is the following:  

* ``scapy_layers.py``: files which allows the analysis of some protocols not supported by Scapy.  
* ``script_extraction.py``: loads a ``.pcap`` file and retrieves information about each packet (size, headers, arrival time, ...). The data is then exported as several .csv files. The "Google Home" version is specific to Google Home data. This version allows to browse several files in the same folder.  
* ``script_reducer_*.py``: aggregates all the ``.csv`` files formed after using the script ``script_extraction.py``. The "Google Home" version is specific to Google Home data.  
* ``script_flow_extraction.py``: identifies the flows.  
* ``script_feature_engineering.py``: In a first step, the script extracts new characteristics such as the flow on a jumping or sliding window, the time difference between two successive packets, ... In a second step, the categorical values are transformed into numerical values. A ``log10`` transformation is applied on the time features. The "Google Home" version is specific to Google Home data.  

The data from the LoRaWAN network did not need any processing. Only the extraction of new features (script ``script_feature_engineering.py``) had to be done.  



Models
------------

This folder contains the models used for generation. The models are named as follows:  

* ``LSTM``: Long Short-Term Memory.   
* ``GMM``: Gaussian Mixture Model.  
* ``VAE``: Variational AutoEncoder.  



Samples
------------

The folder contains sample data used as input to scripts or obtained as output. Here is the list 
of the files present and the associated information:  

* 



Requirements
------------

* Python 3.6.0  
* Keras  2.0.5  
* TensorFlow 2.0  
* Numpy 1.14.3  
* Pandas 0.22.0  
* Scapy 2.4.3  
* Scapy_ssl_tls 2.0.0  



Updates
-------

* Version 0.0.1  



Authors
-------

* **Fabien Meslet-Millet**  


Contributors
------------

*

LICENSE
-------

See the file "LICENSE" for information.

