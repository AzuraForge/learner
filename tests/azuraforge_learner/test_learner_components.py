# ========== GÜNCELLENECEK DOSYA: tests\azuraforge_learner\test_learner_components.py ==========
import pytest
import numpy as np

from azuraforge_learner import Learner, Sequential, Linear, ReLU, MSELoss, SGD

def test_learner_fit_simple_regression():
    X_train = np.array([[-1.0], [0.0], [1.0], [2.0]], dtype=np.float32)
    y_train = np.array([[-1.0], [1.0], [3.0], [5.0]], dtype=np.float32)
    
    model = Sequential(Linear(1, 1))
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.1)
    learner = Learner(model, criterion, optimizer)
    
    initial_loss = learner.evaluate(X_train, y_train)['val_loss']
    
    learner.fit(X_train, y_train, epochs=30)
    
    final_loss = learner.history['loss'][-1]
    
    print(f"Initial Loss: {initial_loss}, Final Loss: {final_loss}")
    assert final_loss < initial_loss / 5

def test_sequential_model_forward_pass():
    model = Sequential(Linear(2, 4), ReLU(), Linear(4, 1))
    from azuraforge_core import Tensor
    
    input_tensor = Tensor(np.random.randn(10, 2))
    output_tensor = model(input_tensor)
    
    assert output_tensor.data.shape == (10, 1)