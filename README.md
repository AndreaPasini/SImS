# SImS (Semantic Image Summarization)

This repository refers to the article *Semantic Image Collection Summarization with Frequent Subgraph Mining*.<br/>
Authors: Andrea Pasini, Elena Baralis, Politecnico di Torino.

Description of the program entry points:

**main_position_classifier.py**
  - Train/validate the relative-position classifier on our position dataset
  
**main_PRS.py**:
  - Build scene graphs (with object positions) for COCO (train, val and panoptic predictions)
  - Generate the Pairwise Relationship Summary (PRS) from scene graphs
  
**main_SGS.py**
  - Apply frequent subgraph mining to the scene graphs, to derive the Scene Graph Summary (SGS)
  - Reproduce the different experimental configuration provided in our white paper
  - Show frequent graphs with charts
  
  **main_sims.py**
  - The complete SImS pipeline (designed for COCO, but with minor changes can be applied to other datasets), including scene graph computation, PRS and SGS building.
  
  **main_competitors.py**
  - This file provides the implementation of the KMedoids technique, used as baseline.
  
  Our labeled COCO subset for training the relative position classifier and the generated summaries can be found at:
  https://drive.google.com/file/d/1qZNZyAgGWkUrzFrpZaOn9-tEYWZKPo-u/view?usp=sharing


