import pytest
import numpy as np
from azuraforge_core import Tensor
from azuraforge_learner import Learner, Sequential, Linear, LSTM, ReLU, MSELoss, SGD, Adam

# --- Learner ve Optimizatör Testleri ---

@pytest.mark.parametrize("optimizer_class, lr", [(SGD, 0.1), (Adam, 0.01)])
def test_learner_fit_simple_regression(optimizer_class, lr):
    """Learner'ın hem SGD hem de Adam ile basit bir regresyon problemini çözebildiğini test eder."""
    X_train = np.array([[-1.0], [0.0], [1.0], [2.0]], dtype=np.float32)
    # y = 2x + 1
    y_train = np.array([[-1.0], [1.0], [3.0], [5.0]], dtype=np.float32)
    
    model = Sequential(Linear(1, 1))
    criterion = MSELoss()
    optimizer = optimizer_class(model.parameters(), lr=lr)
    learner = Learner(model, criterion, optimizer)
    
    # Eğitmeden önceki kayıp
    initial_loss = learner.evaluate(X_train, y_train)['val_loss']
    
    learner.fit(X_train, y_train, epochs=50)
    
    # Eğitimden sonraki kayıp
    final_loss = learner.history['loss'][-1]
    
    print(f"Optimizer: {optimizer_class.__name__}, Initial Loss: {initial_loss:.4f}, Final Loss: {final_loss:.4f}")
    
    # Kaybın anlamlı bir şekilde düştüğünü kontrol et
    assert final_loss < initial_loss / 10

# --- Katman Testleri ---

def test_sequential_model_forward_pass():
    model = Sequential(Linear(10, 32), ReLU(), Linear(32, 1))
    input_tensor = Tensor(np.random.randn(5, 10)) # (batch_size, input_features)
    output_tensor = model(input_tensor)
    
    assert output_tensor.data.shape == (5, 1)

def test_lstm_layer_forward_pass_shape():
    """LSTM katmanının doğru çıktı şeklini verdiğini test eder."""
    batch_size = 4
    seq_length = 10
    input_size = 5
    hidden_size = 20

    lstm_layer = LSTM(input_size=input_size, hidden_size=hidden_size)
    
    # Girdi: (batch, sequence, features)
    input_data = np.random.randn(batch_size, seq_length, input_size)
    input_tensor = Tensor(input_data)
    
    output_tensor = lstm_layer.forward(input_tensor)

    # forward() metodunun tüm zaman adımlarını döndürdüğünü varsayıyoruz
    assert output_tensor.data.shape == (batch_size, seq_length, hidden_size), \
        f"LSTM output shape is wrong. Expected {(batch_size, seq_length, hidden_size)}, got {output_tensor.data.shape}"

def test_lstm_in_sequential_model():
    """LSTM katmanının bir Sequential model içinde doğru çalıştığını test eder."""
    batch_size = 4
    seq_length = 10
    input_size = 5
    hidden_size = 20
    
    model = Sequential(
        LSTM(input_size=input_size, hidden_size=hidden_size),
        # LSTM çıktısı (batch, seq, hidden) olduğu için, 
        # sadece son zaman adımını alıp Linear katmana vermek gerekir.
        # Ancak bizim LSTM'imiz zaten sadece son adımı döndürüyor.
        Linear(hidden_size, 1)
    )
    
    input_data = np.random.randn(batch_size, seq_length, input_size)
    input_tensor = Tensor(input_data)
    
    output_tensor = model(input_tensor)

    # Çıktı: (batch_size, output_features)
    assert output_tensor.data.shape == (batch_size, 1), \
        f"Model output shape is wrong. Expected {(batch_size, 1)}, got {output_tensor.data.shape}"