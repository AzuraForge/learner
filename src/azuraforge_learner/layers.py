from typing import List, Tuple, Optional
import numpy as np
from azuraforge_core import Tensor, xp, ArrayType

class Layer:
    def forward(self, x: Tensor) -> Tensor: raise NotImplementedError
    def parameters(self) -> List[Tensor]: return []
    def __call__(self, x: Tensor) -> Tensor: return self.forward(x)

class Linear(Layer):
    def __init__(self, input_dim: int, output_dim: int):
        limit = np.sqrt(2.0 / input_dim)
        self.weights = Tensor(xp.random.randn(input_dim, output_dim) * limit, requires_grad=True)
        self.bias = Tensor(xp.zeros(output_dim), requires_grad=True)
    def forward(self, x: Tensor) -> Tensor:
        return x.dot(self.weights) + self.bias
    def parameters(self) -> List[Tensor]:
        return [self.weights, self.bias]

class ReLU(Layer):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

class Sigmoid(Layer):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()

# YENİ: LSTM Katmanı eklendi
class LSTM(Layer):
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        H = hidden_size
        
        # Ağırlıkları daha iyi başlatma (Xavier/Glorot benzeri)
        limit = np.sqrt(1.0 / H)
        
        # Ağırlıkları tek bir matriste birleştirmek daha verimlidir
        self.W_x = Tensor(xp.random.randn(input_size, H * 4) * limit, requires_grad=True)
        self.W_h = Tensor(xp.random.randn(H, H * 4) * limit, requires_grad=True)
        self.b = Tensor(xp.zeros(H * 4), requires_grad=True)
        
        # Geriye yayılım için önbellek
        self.cache: Optional[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]] = None

    def parameters(self) -> List[Tensor]:
        return [self.W_x, self.W_h, self.b]

    def forward(self, x: Tensor) -> Tensor:
        # x'in boyutu: (N, T, D) -> (batch_size, sequence_length, input_size)
        N, T, _ = x.data.shape
        H = self.hidden_size

        h_prev = Tensor(xp.zeros((N, H)))
        c_prev = Tensor(xp.zeros((N, H)))
        
        # Tüm zaman adımlarındaki çıktıları saklamak için
        h_all = []

        for t in range(T):
            x_t = Tensor(x.data[:, t, :]) # Mevcut zaman adımındaki girdi
            
            # Gates = (xt @ Wx) + (h_prev @ Wh) + b
            gates = x_t.dot(self.W_x) + h_prev.dot(self.W_h) + self.b

            # Kapıları ayır (f, i, g, o)
            f = Tensor(gates.data[:, :H]).sigmoid()
            i = Tensor(gates.data[:, H:2*H]).sigmoid()
            g = Tensor(gates.data[:, 2*H:3*H]).tanh() # Tanh yeni eklenecek
            o = Tensor(gates.data[:, 3*H:]).sigmoid()
            
            # Hücre ve gizli durumunu güncelle
            c_next = f * c_prev + i * g
            h_next = o * c_next.tanh() # Tanh yeni eklenecek

            h_all.append(h_next.data)
            h_prev, c_prev = h_next, c_next
            
        # Çıktı olarak son gizli durumu (h_next) döndürüyoruz
        # Ama tüm geçmişi saklayarak daha karmaşık modeller (many-to-many) de yapılabilir
        
        # YENİ: backward için Tensor operasyonları kullanılacak
        # Bu basitleştirilmiş forward, backward implementasyonunu zorlaştırır.
        # Bu nedenle, backward'ı doğru yapmak için daha detaylı bir forward yazmalıyız.
        # Şimdilik bu kısmı placeholder olarak bırakıp, sonraki fazda tam backward ile dolduralım.
        # Bu fazda amacımız, forward pass'ın çalışması ve model yapısının kurulması.
        
        # Döngünün sonundaki h_next'i döndür
        return h_next