import logging
import os

import numpy as np
import numpy.typing as npt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from typing import TYPE_CHECKING

from ..utils import MODEL_FOLDER

if TYPE_CHECKING:
    from .features import F1FeatureProcessor


def set_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class F1Dataset(Dataset):
    def __init__(self, X: torch.FloatTensor, y: torch.FloatTensor, sample_weights: torch.FloatTensor) -> None:
        self.X = X
        self.y = y
        self.sample_weights = sample_weights

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.sample_weights is not None:
            return {"X": self.X[idx], "y": self.y[idx], "weight": self.sample_weights[idx]}
        return {"X": self.X[idx], "y": self.y[idx]}


class DriverAttentionLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.3, use_batch_norm: bool = True) -> None:
        super().__init__()

        if input_size <= 0:
            raise ValueError(f"Input Size MUST be POSITIVE, got {input_size}")
        if hidden_size <= 0:
            raise ValueError(f"Hidden Size MUST be POSITIVE, got {hidden_size}")

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.use_batch_norm = use_batch_norm

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )

        if use_batch_norm:
            self.norm1 = nn.BatchNorm1d(hidden_size)
            self.norm2 = nn.BatchNorm1d(hidden_size // 2)
        else:
            self.norm1 = nn.GroupNorm(min(8, hidden_size), hidden_size)
            self.norm2 = nn.GroupNorm(min(4, hidden_size // 2), hidden_size // 2)

        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),

            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),

            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    if param.dim() >= 2:
                        nn.init.orthogonal_(param)
                    else:
                        nn.init.uniform_(param, -0.1, 0.1)
                else:
                    if param.dim() >= 2:
                        nn.init.xavier_normal_(param)
                    else:
                        nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)

        lstm_out, _ = self.lstm(x, (h0, c0))
        attn_weights = self.attention(lstm_out)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        out = self.fc_layers[0](context)

        out = self.norm1(out)

        out = F.relu(out)
        out = F.dropout(out, p=self.dropout_rate, training=self.training)

        out = self.fc_layers[3](out)

        out = self.norm2(out)

        out = F.relu(out)
        out = F.dropout(out, p=self.dropout_rate, training=self.training)

        out = self.fc_layers[6](out)

        return out


class SpearmanLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)

        n = pred.size(0)
        if n <= 1:
            return torch.tensor(1.0, device=pred.device, requires_grad=True)

        pred_ranks = self._differentiable_rank(pred)
        target_ranks = self._differentiable_rank(target)

        pred_mean = torch.mean(pred_ranks)
        target_mean = torch.mean(target_ranks)

        pred_diff = pred_ranks - pred_mean
        target_diff = target_ranks - target_mean

        cov = torch.mean(pred_diff * target_diff)
        pred_std = torch.sqrt(torch.mean(pred_diff ** 2) + 1e-8)
        target_std = torch.sqrt(torch.mean(target_diff ** 2) + 1e-8)

        corr = cov / (pred_std * target_std)
        loss = 1 - corr

        return loss

    def _differentiable_rank(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)

        one_hot = torch.zeros(n, n, device=x.device)
        for i in range(n):
            one_hot[i] = torch.sigmoid((x - x[i].unsqueeze(0)) * 10.0)

        rank = torch.sum(one_hot, dim=1) + 1
        return rank


class PositionRankingLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)

        n = pred.size(0)
        if n <= 1:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        loss = 0.0
        count = 0

        for i in range(n):
            for j in range(i + 1, n):
                if target[i] < target[j]:

                    loss += torch.relu(self.margin - (pred[j] - pred[i]))
                    count += 1
                elif target[i] > target[j]:

                    loss += torch.relu(self.margin - (pred[i] - pred[j]))
                    count += 1

        if count > 0:
            return loss / torch.tensor(count, device=pred.device, dtype=torch.float)
        return torch.tensor(0.0, device=pred.device, requires_grad=True)


class CombinedRaceLoss(nn.Module):
    def __init__(self, rank_weight: float = 0.5, spearman_weight: float = 0.5):
        super().__init__()
        self.rank_loss = PositionRankingLoss()
        self.spearman_loss = SpearmanLoss()
        self.rank_weight = rank_weight
        self.spearman_weight = spearman_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ranking = self.rank_loss(pred, target)
        spearman = self.spearman_loss(pred, target)

        return (self.rank_weight * ranking +
                self.spearman_weight * spearman)


class F1RacePredictor:

    MODEL_NAME = "F1_Model"

    def __init__(self, use_saved_model: bool = True, seed: int = 42) -> None:
        self.use_saved_model = use_saved_model
        self.model = None
        self.feature_processor = None
        self.sequence_length = 5
        self.hidden_size = 256
        self.num_layers = 3
        self.batch_size = 32
        self.num_epochs = 100
        self.learning_rate = 0.001
        self.seed = seed
        set_seeds(seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout_rate = 0.3

        os.makedirs(MODEL_FOLDER, exist_ok=True)

    def train(
        self,
        features_processor: 'F1FeatureProcessor',
        X_train: torch.FloatTensor,
        y_train: torch.FloatTensor,
        sample_weights_train: torch.FloatTensor,
        X_val: torch.FloatTensor,
        y_val: torch.FloatTensor,
        sample_weights_val: torch.FloatTensor
    ) -> None:
        set_seeds(self.seed)
        self.feature_processor = features_processor

        if self.use_saved_model and os.path.exists(f"{MODEL_FOLDER}/{F1RacePredictor.MODEL_NAME}.pt"):
            self.load_model()
            return

        if len(X_train) == 0 or len(y_train) == 0:
            return

        effective_batch_size = min(self.batch_size, len(X_train))
        if effective_batch_size <= 1:
            logging.warning(f"Only {len(X_train)} Training Samples Available - Using Group Norm Instead of Batch Norm")

            effective_batch_size = 1

        train_dataset = F1Dataset(X_train, y_train, sample_weights_train)
        val_dataset = F1Dataset(X_val, y_val, sample_weights_val)

        train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=effective_batch_size, drop_last=True)

        input_size = X_train.shape[2]

        self.model = DriverAttentionLSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=1,
            dropout=self.dropout_rate,
            use_batch_norm=(effective_batch_size > 1)
        ).to(self.device)

        mse_criterion = nn.MSELoss()
        ranking_criterion = CombinedRaceLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        best_val_loss = float('inf')
        early_stop_counter = 0
        early_stop_patience = 15

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0
            batch_count = 0

            for batch in train_loader:
                batch_X = batch["X"].to(self.device)
                batch_y = batch["y"].to(self.device).view(-1, 1)
                batch_weights = batch["weight"].to(self.device).view(-1, 1)

                optimizer.zero_grad()
                outputs = self.model(batch_X)

                mse_loss = mse_criterion(outputs, batch_y)
                rank_loss = ranking_criterion(outputs.view(-1), batch_y.view(-1))
                loss = mse_loss + 0.5 * rank_loss

                weighted_loss = loss * batch_weights.mean()

                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += weighted_loss.item()
                batch_count += 1

            self.model.eval()
            val_loss = 0.0
            val_batch_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch_X = batch["X"].to(self.device)
                    batch_y = batch["y"].to(self.device).view(-1, 1)
                    batch_weights = batch["weight"].to(self.device).view(-1, 1)

                    outputs = self.model(batch_X)

                    mse_loss = mse_criterion(outputs, batch_y)
                    rank_loss = ranking_criterion(outputs.view(-1), batch_y.view(-1))
                    loss = mse_loss + 0.5 * rank_loss

                    weighted_loss = loss * batch_weights.mean()
                    val_loss += weighted_loss.item()
                    val_batch_count += 1

            avg_train_loss = train_loss / max(1, batch_count)
            avg_val_loss = val_loss / max(1, val_batch_count)
            scheduler.step(avg_val_loss)

            if (epoch + 1) % 5 == 0 or epoch < 5:
                logging.info(f'Epoch {epoch + 1:02}/{self.num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}')

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stop_counter = 0
                self.save_model()
            else:
                early_stop_counter += 1

            if early_stop_counter >= early_stop_patience:
                logging.info(f'Early Stop Triggered at Epoch #{epoch + 1}')
                break

        self.load_model()

    def predict(
        self,
        X_test: torch.Tensor,
        test_drivers: list[str],
        qualifying_positions: list[int] | None = None
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        set_seeds(self.seed)

        if self.model is None:
            raise ValueError("Model Not Trained - Cannot Predict")

        if len(X_test) == 0:
            raise ValueError("No Valid Test Data Found for Prediction")

        self.model.eval()
        with torch.no_grad():
            X_test = X_test.to(self.device)
            predicted_positions = self.model(X_test).cpu().numpy().flatten()

        if qualifying_positions:
            adjusted_positions = self._position_predictor_adjustment(
                predicted_positions, test_drivers, qualifying_positions
            )
        else:
            adjusted_positions = self._position_predictor_adjustment(
                predicted_positions, test_drivers
            )

        return predicted_positions, adjusted_positions

    def _position_predictor_adjustment(
        self,
        predicted_positions: npt.NDArray[np.float64],
        test_drivers: list[str],
        qualifying_positions: list[int] | None = None
    ):
        adjusted_positions = predicted_positions.copy()

        grid_positions = {}
        if qualifying_positions is not None:
            for i, driver in enumerate(test_drivers):
                grid_positions[driver] = qualifying_positions[i]

        top_tier_drivers = getattr(self.feature_processor, 'top_drivers', [])
        mid_tier_drivers = getattr(self.feature_processor, 'mid_tier_drivers', [])

        position_boundaries = getattr(self.feature_processor, 'position_boundaries', {})

        for i, driver in enumerate(test_drivers):
            original_prediction = predicted_positions[i]

            if driver in top_tier_drivers:
                if original_prediction > 10:
                    adjusted_positions[i] = (original_prediction + 7) / 2
            elif driver in mid_tier_drivers:
                if original_prediction < 5:
                    adjusted_positions[i] = (original_prediction + 7) / 2
                elif original_prediction > 12:
                    adjusted_positions[i] = (original_prediction + 10) / 2

            if driver in position_boundaries:
                q1, q3 = position_boundaries[driver]

                if original_prediction < q1 - 3:
                    adjusted_positions[i] = (original_prediction * 0.3 + q1 * 0.7)
                elif original_prediction > q3 + 3:
                    adjusted_positions[i] = (original_prediction * 0.3 + q3 * 0.7)
                elif original_prediction < q1 - 1:
                    adjusted_positions[i] = (original_prediction * 0.6 + q1 * 0.4)
                elif original_prediction > q3 + 1:
                    adjusted_positions[i] = (original_prediction * 0.6 + q3 * 0.4)

            if driver in grid_positions:
                grid_pos = grid_positions[driver]

                if grid_pos <= 3:
                    grid_weight = 0.5
                    adjusted_positions[i] = (adjusted_positions[i] * (1 - grid_weight) + grid_pos * grid_weight)
                elif grid_pos <= 10:
                    grid_weight = 0.3
                    adjusted_positions[i] = (adjusted_positions[i] * (1 - grid_weight) + grid_pos * grid_weight)

        adjusted_positions = np.maximum(adjusted_positions, 1.0)

        return adjusted_positions

    def save_model(self) -> None:
        assert self.model is not None, "Model NOT Trained - Cannot Save"
        torch.save(self.model.state_dict(), f"{MODEL_FOLDER}/{F1RacePredictor.MODEL_NAME}.pt")

    def load_model(self) -> None:
        assert self.feature_processor is not None, "Feature Processor MUST be Set Before Loading Model"

        if not os.path.exists(f"{MODEL_FOLDER}/{F1RacePredictor.MODEL_NAME}.pt"):
            logging.error(f"Model File Not Found: {MODEL_FOLDER}/{F1RacePredictor.MODEL_NAME}.pt")
            return

        driver_size = len(self.feature_processor.driver_encoder.categories_[0]) # type: ignore
        constructor_size = len(self.feature_processor.constructor_encoder.categories_[0]) # type: ignore
        circuit_size = len(self.feature_processor.circuit_encoder.categories_[0]) # type: ignore
        numerical_size = self.feature_processor.feature_scaler.n_features_in_ # type: ignore
        input_size = driver_size + constructor_size + circuit_size + numerical_size

        saved_state_dict = torch.load(f"{MODEL_FOLDER}/{F1RacePredictor.MODEL_NAME}.pt", map_location=self.device)

        uses_batch_norm = any('running_mean' in key for key in saved_state_dict.keys())

        logging.info(f"Loading Saved Model with BatchNorm: {uses_batch_norm}")

        self.model = DriverAttentionLSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=1,
            dropout=self.dropout_rate,
            use_batch_norm=uses_batch_norm
        ).to(self.device)

        if not uses_batch_norm:
            filtered_state_dict = {k: v for k, v in saved_state_dict.items() if not any(x in k for x in ['running_mean', 'running_var', 'num_batches_tracked'])}

            try:
                self.model.load_state_dict(filtered_state_dict, strict=False)
                logging.info("Successfully Loaded Model with Filtered State Dict")
            except Exception as E:
                logging.error(f"Error Loading Model with Filtered State Dict:", exc_info=E)

                self.model = DriverAttentionLSTM(
                    input_size=input_size,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    output_size=1,
                    dropout=self.dropout_rate,
                    use_batch_norm=True
                ).to(self.device)
                self.model.load_state_dict(saved_state_dict)
                logging.info("Fallback to Original Model with BatchNorm")
        else:
            self.model.load_state_dict(saved_state_dict)

        self.model.eval()
