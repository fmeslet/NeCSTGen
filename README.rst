======================================================================================
NeCSTGen: An approach For Realistic Network Traffic Generation Using Deep Learning
======================================================================================



Summary
------------

NeCSTGen (Network Clustering Sequential Traffic Generation) is an approach composed of multiple models based on Deep Learning architecture for network traffic generation. The model reproduces the original behavior at the packet, flow and aggregate levels. This work is published in `IEEE GLOBECOM 2022 <https://ieeexplore.ieee.org/document/10000731>`.



Processing
------------

The folder ``processing`` includes all the files that have allowed to process the data and to transform them to make the modeling.

Il y a trois étapes lors du traitement des données :

* Data extraction from ``.pcap`` files, contained in the folder ``processing/packets/`` ;
* Flows identification, contained in the folder ``processing/flows/`` ;
* Features enginnering, contained in the folder ``scripts/processing/packets/`` .

Script ``scapy_layers.py`` allows the analysis of some protocols not supported by Scapy such as: IRC, IMAP, SMTP, POP, SIP, SSH, Telnet, FTP. 


Data extraction
^^^^^^^^^^^^

The files present as well as their order of use is the following: 

* ``script_extraction_*.py`` loads a ``.pcap`` file and retrieves information about each packet (size, headers, arrival time, ...). The data is then exported as several ``.csv`` files named ````. The "Google Home" version is specific to Google Home data. This version allows to browse several files in the same folder. 
* ``script_reducer_*.py``: aggregates all the ``.csv`` files formed after using the script ``script_extraction.py``. The "Google Home" version is specific to Google Home data. Les fichiers, une fois renommé, sont appellé ``df_week{1, 3}_DAY.csv``.

After this step we get files like, for example: ``df_week3_thursday.csv``. Each file contains all the packets with their characteristics for each day and each week.


Flows identification
^^^^^^^^^^^^

Script ``script_flow_extraction.py`` identifies the flows. Protocols that have no flow (no source or destination port) are identified from their source and destination MAC address. In this case, the purpose is to be able to identify communications between different devices.


Features enginnering
^^^^^^^^^^^^

The features enginnerring step consists of two sub-steps:

* Feature extraction: consists in extracting new characteristics such as: packet size, header size, throughput in bytes/s, ... .
* Feature processing: the data are transformed to facilitate learning and modeling.  

The files present as well as their order of use is the following:  
 
* ``script_feature_extraction.py``: In a first step, the script extracts new features such as the flow on a jumping or sliding window, the time difference between two successive packets, ... . The "Google Home" and LoRaWAN version are specific to these data. 
* ``script_feature_processing.py``:  Transform the feature for the modeeling. The categorical values are transformed into numerical values. A ``log10`` transformation is applied on the time features. All the feature are standardize between 0 and 1. The "Google Home" and LoRaWAN version are specific to these data. 



Training
------------

The files are present in the ``training`` folder for each associated level. The levels are as follows:  

* Packet level  
* Flow level  
* Agregate level  


Packet level
^^^^^^^^^^^^

We sample a cluster identified in the latent space thanks to the GMM (Gaussian Mixture Model) and we use the VAE (Variational AutoEncoder) decoder to reconstruct the parameters of the packet. The following scripts are to be used:  

* ``script_vae_training.py``: allows the learning of the VAE (Variational AutoEncoder).
* ``script_gmm_training.py``: shows how to train the GMM (Gaussian Mixture Model).  


Flow level
^^^^^^^^^^^^

The files used to trained the LSTM (Long Short-Term Memory) model for flow generation are the following: 

* ``script_lstm_scapy_flow_connect_training.py``: allows the learning of the LSTM (Long Short-Term Memory) model for the generation of a flow in connected mode. For example, applications using TCP (Transport Control Protocol), the start and end structure of the flow will be taken into account.  
* ``script_lstm_scapy_flow_no_connect_training.py``: allows the learning of the LSTM (Long Short-Term Memory) model for the generation of a flow in unconnected mode. For example, a communication between two devices in UDP (User Datagram Protocol) mode.  
* ``script_lstm_scapy_no_flow_*_training.py``: allows the learning of the LSTM (Long Short-Term Memory) model for the generation of a set of packets without flow information. The "LoRaWAN" version is specific to LoRaWAN data.  


Agregate level
^^^^^^^^^^^^^^

* ``script_flows_generation_scapy.py``: shows how to train the GMM (Gaussian Mixure Model) allowing the generation of the characteristics of a flow.  



Generation
------------

The generation files are present in the ``inference`` folder for each associated level. The levels are the following: 

* Packet level   
* Flow level  
* Agregate level   


Packet level
^^^^^^^^^^^^

We sample a cluster identified in the latent space thanks to the GMM (Gaussian Mixture Model) and we use the VAE (Variational AutoEncoder) model decoder to reconstruct the parameters of the packet. The following scripts are to be used:  

* ``script_packet_generation.py``: shows how to generate a packet with the GMM (Gaussian Mixture Model) and the VAE (Variational Auto-Encoder).  


Flow level
^^^^^^^^^^^^

Each file contains a function ``gen_pcap`` within this function the pieces of code allowing the generation of the ``.pcap`` file are commented. The files are the following:  

* ``script_packet_generation_scapy_flow_connect_*.py``: allows the generation of flows in connected mode. For example, applications using TCP (Transport Control Protocol), the start and end structure of the flow will be taken into account. The "Google Home" version is specific to Google Home data.  
* ``script_packet_generation_scapy_flow_no_connect_*.py``: allows the generation of a flow in unconnected mode. For example, a communication between two devices in UDP (User Datagram Protocol) mode. The "Google Home" version is specific to Google Home data.  
* ``script_packet_generation_scapy_no_flow_*py``: allows the generation of a set of packets without flow information. The "LoRaWAN" version is specific to LoRaWAN data.  

The generation of a set of flows can be done by following the dynamics present in the original data (a dynamic present over a particular period) or by using an aggregate of generated flows.  


Agregate level
^^^^^^^^^^^^^^

* ``script_flows_generation_scapy.py``: allows the generation of a set of flows but not the associated packets. A flow level generation will have to be used to be able to generate a set of packets.  



Notebooks
------------

The notebook in the directory is the one used to generate the graphs to analyze the traffic generated.    



Models
------------

This folder contains the models used for generation. The models are named as follows:  

* ``LSTM``: Long Short-Term Memory.   
* ``GMM``: Gaussian Mixture Model.  
* ``VAE``: Variational AutoEncoder.  



Samples
------------

The folder contains sample data used as input to scripts or obtained as output. Here is the list of the files present and the associated information:  

* 



Requirements
------------

* Python 3.6.0  
* TensorFlow 2.4.1  
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

