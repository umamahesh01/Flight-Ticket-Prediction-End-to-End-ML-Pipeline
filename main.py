import pandas as pd
import yaml
from src.data.data_loader import DataLoader
from src.preprocessing.missing_value_handler import FillMissingValuesStrategy
from src.preprocessing.feature_engineering.encoding import LabelEncoding
from src.preprocessing.feature_engineering.scaler import Scaling
from src.model.model_trainer import LinearModel
from src.model.model_eval import evaluate
from sklearn.model_selection import train_test_split

# Loading Config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Loading Dataset
loader = DataLoader(config['data_path'])
df = loader.load()

if 'Unnamed: 0' in df.columns:
    df.drop(columns='Unnamed: 0', inplace=True)

# Filling Missing values
handler = FillMissingValuesStrategy(num_method='mean', cat_method='mode')
df = handler.handle(df)

# Encoding
encoder = LabelEncoding()
df = encoder.encode(df, config['features'])

# Scaling
scal = Scaling()
df = scal.scale(df, config['features'])

# train, test split
x = df[config['features']]
y = df[config['target_column']]

# Train, Test Split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Training
trainer = LinearModel()
trainer.train(X_train, y_train)
trainer.save(config['model_path'])




# Predict on test set
y_pred = trainer.model.predict(X_test)
score = evaluate.eval(y_test, y_pred)
print(f'Score is: {score}')


'''
import pickle
model = pickle.load(open('artifacts/model.pkl', 'rb'))
'''