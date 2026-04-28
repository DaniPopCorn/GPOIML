import pandas as pd
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from momentfm import MOMENTPipeline

# ==========================================
# 1. CONFIGURATION AND DATA LOADING
# ==========================================
CSV_FILE_IN = sys.argv[1]
CSV_FILE_OUT = sys.argv[2]

# debug CSV_FILE_IN = "C:/Users/Filippo/Desktop/scuola/5DS/GPOI/Data.csv";
# debug CSV_FILE_OUT = "C:/Users/Filippo/Desktop/scuola/5DS/GPOI/Out.csv";

# Use latin1 encoding to handle special characters from Excel/Windows exports
df = pd.read_csv(CSV_FILE_IN, encoding='latin1').iloc[:-1]

# Model Parameters
CONTEXT_SIZE = 5   # Past years to observe
SEQ_LEN = 512      # Input length required by the MOMENT model
TARGET_YEAR = 2026 

last_year_in_csv = int(df['Anno'].max())
HORIZON = TARGET_YEAR - last_year_in_csv

# Data Cleanup: Remove the 'Year' column for feature processing
features_df = df.drop(columns=['Anno'])
N_CHANNELS = features_df.shape[1]
numpy_data = features_df.values

# ==========================================
# 2. DATASET AND DATALOADER
# ==========================================
class MacroDataDataset(Dataset):
    def __init__(self, data, context_size, forecast_horizon, seq_len):
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(data)
        self.context_size = context_size
        self.forecast_horizon = forecast_horizon
        self.seq_len = seq_len

    def __len__(self):
        # Calculate available windows for training
        return len(self.data) - self.context_size - self.forecast_horizon + 1

    def __getitem__(self, idx):
        x_real = self.data[idx : idx + self.context_size]
        y_real = self.data[idx + self.context_size : idx + self.context_size + self.forecast_horizon]
        
        # Padding logic to reach SEQ_LEN (512)
        x_padded = np.zeros((self.seq_len, x_real.shape[1]))
        x_padded[-self.context_size:] = x_real
        mask = np.zeros(self.seq_len)
        mask[-self.context_size:] = 1 # Mask identifies real data vs padding
        
        return (torch.tensor(x_padded, dtype=torch.float32).transpose(0, 1), 
                torch.tensor(mask, dtype=torch.float32), 
                torch.tensor(y_real, dtype=torch.float32).transpose(0, 1))

dataset = MacroDataDataset(numpy_data, CONTEXT_SIZE, HORIZON, SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=True)

# ==========================================
# 3. MODEL LOADING AND FINE-TUNING
# ==========================================
print(f"Initializing for target year {TARGET_YEAR}...")
model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large", 
    model_kwargs={"task_name": "forecasting", "forecast_horizon": HORIZON, "n_channels": N_CHANNELS}
)
model.init()

# Linear Probing: Freeze the foundation model backbone and only train the prediction head
for name, param in model.named_parameters():
    if "head" not in name: 
        param.requires_grad = False

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
criterion = nn.MSELoss()

model.train()
print("Starting training...")
epochNum = 30
for epoch in range(epochNum):
    total_loss = 0
    for batch_x, batch_mask, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(x_enc=batch_x, input_mask=batch_mask)
        loss = criterion(outputs.forecast, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochNum} - Loss: {total_loss/len(dataloader):.4f}")

# ==========================================
# 4. METRICS CALCULATION (Accuracy, Precision, Recall, Confusion Matrix)
# ==========================================
model.eval()
all_preds, all_trues = [], []

with torch.no_grad():
    for batch_x, batch_mask, batch_y in dataloader:
        out = model(x_enc=batch_x, input_mask=batch_mask).forecast
        all_preds.append(out.numpy())
        all_trues.append(batch_y.numpy())

# Transform results into binary trends: 1 = Growth (relative to mean), 0 = Decline/Stagnation
y_pred_bin = (np.concatenate(all_preds) > 0).astype(int).flatten()
y_true_bin = (np.concatenate(all_trues) > 0).astype(int).flatten()

print("\n--- PERFORMANCE REPORT (Trend Classification) ---")
print(f"Accuracy:  {accuracy_score(y_true_bin, y_pred_bin):.2f}")
print(f"Precision: {precision_score(y_true_bin, y_pred_bin, zero_division=0):.2f}")
print(f"Recall:    {recall_score(y_true_bin, y_pred_bin, zero_division=0):.2f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_true_bin, y_pred_bin))

# ==========================================
# 5. FINAL 2026 PREDICTION AND EXPORT
# ==========================================
print("\nGenerating final report...")
# Scale the most recent data points for inference
latest_data = dataset.scaler.transform(features_df.tail(CONTEXT_SIZE).values)

x_inf = np.zeros((SEQ_LEN, N_CHANNELS))
x_inf[-CONTEXT_SIZE:] = latest_data
mask_inf = np.zeros(SEQ_LEN)
mask_inf[-CONTEXT_SIZE:] = 1

x_t = torch.tensor(x_inf, dtype=torch.float32).transpose(0, 1).unsqueeze(0)
m_t = torch.tensor(mask_inf, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    pred_raw = model(x_enc=x_t, input_mask=m_t).forecast.squeeze(0).transpose(0, 1).numpy()

# Inverse transform to return to original data scale
pred_final = dataset.scaler.inverse_transform(pred_raw)

# Create output DataFrame and save
future_years = [last_year_in_csv + i for i in range(1, HORIZON + 1)]
df_out = pd.DataFrame(pred_final, columns=features_df.columns)
df_out.insert(0, 'Anno', future_years)

df_out.to_csv(CSV_FILE_OUT, index=False, encoding='latin1')
print("--------------------------------------------------")
print("Data saved successfully.")
