import numpy as np
from typing import List
from azuraforge_core import Tensor, xp
from .base import Layer

class Linear(Layer):
    """
    Tam bağlantılı (fully-connected) doğrusal katman.
    Hem 2D (batch, features) hem de 3D (batch, sequence, features) girdileri destekler.
    """
    def __init__(self, input_dim: int, output_dim: int):
        limit = np.sqrt(2.0 / input_dim)
        self.weights = Tensor(xp.random.randn(input_dim, output_dim) * limit, requires_grad=True)
        self.bias = Tensor(xp.zeros(output_dim), requires_grad=True)
        self.input_tensor_3d = False # Girdinin 3D olup olmadığını saklamak için

    def forward(self, x: Tensor) -> Tensor:
        self.input_tensor_3d = x.data.ndim == 3
        
        # Eğer girdi 3D ise (N, T, D_in), 2D'ye (N*T, D_in) dönüştür
        if self.input_tensor_3d:
            N, T, D_in = x.data.shape
            x_reshaped = x.data.reshape(N * T, D_in)
        else:
            x_reshaped = x.data

        # Standart 2D matris çarpımı
        output_reshaped = x_reshaped @ self.weights.data + self.bias.data

        # Eğer orijinal girdi 3D ise, çıktıyı da 3D'ye geri dönüştür (N, T, D_out)
        if self.input_tensor_3d:
            D_out = self.weights.data.shape[1]
            output_data = output_reshaped.reshape(N, T, D_out)
        else:
            output_data = output_reshaped
            
        # Geri yayılım için orijinal tensörleri çocuk olarak ata
        out = Tensor(output_data, _children=(x, self.weights, self.bias), _op="Linear", requires_grad=x.requires_grad)

        def _backward():
            if not out.requires_grad:
                return

            # Gradyanı, ileri geçişteki matris çarpımının şekline uygun hale getir
            grad_reshaped = out.grad.reshape(x_reshaped.shape[0], -1) if self.input_tensor_3d else out.grad

            if self.bias.requires_grad and self.bias.grad is not None:
                self.bias.grad += grad_reshaped.sum(axis=0)

            if self.weights.requires_grad and self.weights.grad is not None:
                self.weights.grad += x_reshaped.T @ grad_reshaped

            if x.requires_grad and x.grad is not None:
                x_grad_reshaped = grad_reshaped @ self.weights.data.T
                # Gradyanı orijinal girdi şekline geri getir
                x.grad += x_grad_reshaped.reshape(x.data.shape)
        
        out._backward = _backward
        return out

    def parameters(self) -> List[Tensor]:
        return [self.weights, self.bias]