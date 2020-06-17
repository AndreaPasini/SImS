# SImS (Semantic Image Summarization)

This repository refers to the article *Semantic Image Collection Summarization with Frequent Subgraph Mining*.<br/>
Authors: xxx, xxx.

Description of the program entry points:

**main_gen_prs.py**:
  - Training the relative-position classifier on our position dataset
  - Building scene graphs (with object positions) for COCO (train, val and panoptic predictions)
  - Generating the Pairwise Relationship Summary (PRS) from training graphs
  
**main_graph_mining.py**
  - Apply frequent subgraph mining to the scene graphs, to derive the Scene Graph Summary (SGS)
  - Reproduce the different experimental configuration provided in our white paper
  
  Our labeled COCO subset for training the relative position classifier and the generated summaries can be found at:
  https://drive.google.com/file/d/1qZNZyAgGWkUrzFrpZaOn9-tEYWZKPo-u/view?usp=sharing

