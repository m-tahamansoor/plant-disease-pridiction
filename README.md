Heirarchy:
plant-disease-classifier/
├── src/
│ ├── __init__.py         
│ ├── config.py           
│ ├── data_loader.py      
│ ├── model.py            
│ └── train.py            
├── notebook_utils/
│ └── visualize.py        
├── models/
│ └── .gitkeep           
├── data/
│ └── new-plant-diseases-dataset/
│     ├── train/
│     └── valid/
| └── test
| └── app.py
└── README.md

to run: & "C:/Program Files/Python311/python.exe" -m src.train

dataset link: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

first we were using MobileNetV2 architechture of CNN which was giving train accuracy of 89% 
in second try we used EfficientNet_B3 which improve train accuracy to 91.88% and give validation accuracy of 97.10%