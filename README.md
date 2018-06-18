# BlurEqualisationTechnique
This is demo of the Blur Eqalisation Technique (BET) presented in [XianSubbarao2006]. 
It uses the _breakfast_  dataset from [WatanabeNayar1998] and POV-Ray generated datasets,
_patio_ and _office_,  presented in [DevernayPujadesVijay2012]. 
The last two datasets feature exact ground truths.

[XianSubbarao2006]: https://doi.org/10.1117/12.688615
[WatanabeNayar1998]: https://doi.org/10.1023/A:1007905828438 
[DevernayPujadesVijay2012]:  https://doi.org/10.1117/12.906209 

## Usage 

Download the repository and use IPython to run:
- `demo_depth_bet.py`, it generates a synthetic dataset; 
- `breakfast_data_bet.py`, it uses the _breakfast_ dataset;
- `povr_data_bet.py`, it can use the _patio_ or _office_ dataset and
it evaluates the method using respective the ground truth. 



