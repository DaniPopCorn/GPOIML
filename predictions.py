import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from momentfm import MOMENTPipeline

# ==========================================
# 1. CONFIGURAZIONE E CARICAMENTO DATI
# ==========================================
CSV_FILE = "C:/Users/Filippo/Desktop/scuola/5DS/GPOI/Dati_ML.csv"
# Utilizziamo latin1 per gestire i caratteri speciali di Excel/Windows
df = pd.read_csv(CSV_FILE, encoding='latin1')

# Parametri modello
CONTEXT_SIZE = 5   # Anni passati da osservare
SEQ_LEN = 512      # Lunghezza richiesta da MOMENT
TARGET_YEAR = 2026 

ultimo_anno_csv = int(df['Anno'].max())
HORIZON = TARGET_YEAR - ultimo_anno_csv

# Pulizia: rimuoviamo l'anno per le feature
features_df = df.drop(columns=['Anno'])
N_CHANNELS = features_df.shape[1]
dati_numpy = features_df.values

# ==========================================
# 2. DATASET E DATALOADER
# ==========================================
class MacroDataDataset(Dataset):
    def __init__(self, data, context_size, forecast_horizon, seq_len):
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(data)
        self.context_size = context_size
        self.forecast_horizon = forecast_horizon
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.context_size - self.forecast_horizon + 1

    def __getitem__(self, idx):
        x_real = self.data[idx : idx + self.context_size]
        y_real = self.data[idx + self.context_size : idx + self.context_size + self.forecast_horizon]
        
        # Padding
        x_padded = np.zeros((self.seq_len, x_real.shape[1]))
        x_padded[-self.context_size:] = x_real
        mask = np.zeros(self.seq_len)
        mask[-self.context_size:] = 1
        
        return (torch.tensor(x_padded, dtype=torch.float32).transpose(0, 1), 
                torch.tensor(mask, dtype=torch.float32), 
                torch.tensor(y_real, dtype=torch.float32).transpose(0, 1))

dataset = MacroDataDataset(dati_numpy, CONTEXT_SIZE, HORIZON, SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=True)

# ==========================================
# 3. CARICAMENTO MODELLO E FINE-TUNING
# ==========================================
print(f"Inizializzo MOMENT per il {TARGET_YEAR}...")
model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large", 
    model_kwargs={"task_name": "forecasting", "forecast_horizon": HORIZON, "n_channels": N_CHANNELS}
)
model.init()

# Congeliamo il backbone, alleniamo solo la testa (Linear Probing)
for name, param in model.named_parameters():
    if "head" not in name: param.requires_grad = False

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
criterion = nn.MSELoss()

model.train()
print("Inizio Addestramento...")
for epoch in range(30):
    total_loss = 0
    for batch_x, batch_mask, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(x_enc=batch_x, input_mask=batch_mask)
        loss = criterion(outputs.forecast, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 5 == 0:
        print(f"Epoca {epoch+1}/30 - Loss: {total_loss/len(dataloader):.4f}")

# ==========================================
# 4. CALCOLO METRICHE (Accuracy, Precision, Recall, CM)
# ==========================================
model.eval()
all_preds, all_trues = [], []

with torch.no_grad():
    for batch_x, batch_mask, batch_y in dataloader:
        out = model(x_enc=batch_x, input_mask=batch_mask).forecast
        all_preds.append(out.numpy())
        all_trues.append(batch_y.numpy())

# Trasformiamo in trend binario: 1 = Crescita rispetto alla media, 0 = Calo/Stasi
y_pred_bin = (np.concatenate(all_preds) > 0).astype(int).flatten()
y_true_bin = (np.concatenate(all_trues) > 0).astype(int).flatten()

print("\n--- PERFORMANCE REPORT (Trend Classification) ---")
print(f"Accuracy:  {accuracy_score(y_true_bin, y_pred_bin):.2f}")
print(f"Precision: {precision_score(y_true_bin, y_pred_bin, zero_division=0):.2f}")
print(f"Recall:    {recall_score(y_true_bin, y_pred_bin, zero_division=0):.2f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_true_bin, y_pred_bin))

# ==========================================
# 5. PREVISIONE FINALE 2026 E SALVATAGGIO
# ==========================================
print("\nGenerazione previsioni finali...")
ultimi_dati = dataset.scaler.transform(features_df.tail(CONTEXT_SIZE).values)

x_inf = np.zeros((SEQ_LEN, N_CHANNELS))
x_inf[-CONTEXT_SIZE:] = ultimi_dati
mask_inf = np.zeros(SEQ_LEN)
mask_inf[-CONTEXT_SIZE:] = 1

x_t = torch.tensor(x_inf, dtype=torch.float32).transpose(0, 1).unsqueeze(0)
m_t = torch.tensor(mask_inf, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    pred_raw = model(x_enc=x_t, input_mask=m_t).forecast.squeeze(0).transpose(0, 1).numpy()

pred_final = dataset.scaler.inverse_transform(pred_raw)

# Creazione DataFrame e salvataggio
anni_futuri = [ultimo_anno_csv + i for i in range(1, HORIZON + 1)]
df_out = pd.DataFrame(pred_final, columns=features_df.columns)
df_out.insert(0, 'Anno', anni_futuri)

df_out.to_csv('Previsioni_MOMENT_2026.csv', index=False, encoding='latin1')
print("--------------------------------------------------")
print("File 'Previsioni_MOMENT_2026.csv' salvato correttamente.")
print("Previsione specifica per il 2026:")
print(df_out[df_out['Anno'] == 2026])