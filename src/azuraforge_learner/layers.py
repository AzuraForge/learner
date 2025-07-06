from typing import List, Tuple, Optional
import numpy as np
from azuraforge_core import Tensor, xp

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

class LSTM(Layer):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        H = hidden_size
        D = input_size
        
        limit = np.sqrt(1.0 / H)
        self.W_x = Tensor(xp.random.randn(D, H * 4) * limit, requires_grad=True)
        self.W_h = Tensor(xp.random.randn(H, H * 4) * limit, requires_grad=True)
        self.b = Tensor(xp.zeros(H * 4), requires_grad=True)
        
        self.cache: Optional[Tuple] = None

    def parameters(self) -> List[Tensor]:
        return [self.W_x, self.W_h, self.b]

    def forward(self, x: Tensor) -> Tensor:
        N, T, D = x.data.shape
        H = self.hidden_size

        h_prev = xp.zeros((N, H))
        c_prev = xp.zeros((N, H))
        
        h_all = xp.zeros((N, T, H))
        c_all = xp.zeros((N, T, H))
        gates_all = xp.zeros((N, T, 4 * H))
        i_all = xp.zeros((N, T, H))
        f_all = xp.zeros((N, T, H))
        o_all = xp.zeros((N, T, H))
        g_all = xp.zeros((N, T, H))

        for t in range(T):
            x_t = x.data[:, t, :]
            gates = x_t @ self.W_x.data + h_prev @ self.W_h.data + self.b.data
            
            i = 1 / (1 + xp.exp(-gates[:, :H]))
            f = 1 / (1 + xp.exp(-gates[:, H:2*H]))
            o = 1 / (1 + xp.exp(-gates[:, 2*H:3*H]))
            g = xp.tanh(gates[:, 3*H:])
            
            c_next = f * c_prev + i * g
            h_next = o * xp.tanh(c_next)

            h_prev, c_prev = h_next, c_next
            
            h_all[:, t, :] = h_next
            c_all[:, t, :] = c_next
            gates_all[:, t, :] = gates
            i_all[:, t, :] = i
            f_all[:, t, :] = f
            o_all[:, t, :] = o
            g_all[:, t, :] = g

        # Tüm ara değerleri içeren ana tensörü oluştur
        out_full_sequence = Tensor(h_all, _children=(x, self.W_x, self.W_h, self.b), _op="lstm", requires_grad=x.requires_grad)
        
        # Geriye yayılım için gerekli tüm ara değerleri sakla
        self.cache = (x.data, h_all, c_all, i_all, f_all, o_all, g_all)

        def _backward():
            # --- GERİ YAYILIM MANTIĞI DEĞİŞMİYOR ---
            # Geri yayılım, gradyanın tüm zaman adımları üzerinden akmasını gerektirir.
            # Biz sadece ileri adımdaki SON çıktıyı değiştiriyoruz, gradyan akışı
            # hala tüm sekansı kullanmalıdır.
            if not out_full_sequence.requires_grad or out_full_sequence.grad is None: return
            assert self.cache is not None, "Cache is not set"
            
            x_data, h_data, c_data, i_data, f_data, o_data, g_data = self.cache
            _N, _T, _D = x_data.shape
            _H = self.hidden_size
            
            dx = xp.zeros_like(x_data)
            dW_x = xp.zeros_like(self.W_x.data)
            dW_h = xp.zeros_like(self.W_h.data)
            db = xp.zeros_like(self.b.data)
            
            dh_next = xp.zeros((_N, _H))
            dc_next = xp.zeros((_N, _H))

            for t in reversed(range(_T)):
                dh = out_full_sequence.grad[:, t, :] + dh_next
                
                dc = dc_next + dh * o_data[:, t, :] * (1 - xp.tanh(c_data[:, t, :])**2)
                
                di = dc * g_data[:, t, :]
                df = dc * (c_data[:, t-1, :] if t > 0 else 0)
                do = dh * xp.tanh(c_data[:, t, :])
                dg = dc * i_data[:, t, :]
                
                d_gates_i = di * i_data[:, t, :] * (1 - i_data[:, t, :])
                d_gates_f = df * f_data[:, t, :] * (1 - f_data[:, t, :])
                d_gates_o = do * o_data[:, t, :] * (1 - o_data[:, t, :])
                d_gates_g = dg * (1 - g_data[:, t, :]**2)
                
                dgates = xp.concatenate((d_gates_i, d_gates_f, d_gates_o, d_gates_g), axis=1)

                x_t = x_data[:, t, :]
                h_prev = h_data[:, t-1, :] if t > 0 else xp.zeros((_N, _H))
                
                dx[:, t, :] = dgates @ self.W_x.data.T
                dh_next = dgates @ self.W_h.data.T
                dc_next = dc * f_data[:, t, :]
                
                dW_x += x_t.T @ dgates
                dW_h += h_prev.T @ dgates
                db += xp.sum(dgates, axis=0)

            if x.requires_grad and x.grad is not None: x.grad += dx
            if self.W_x.requires_grad and self.W_x.grad is not None: self.W_x.grad += dW_x
            if self.W_h.requires_grad and self.W_h.grad is not None: self.W_h.grad += dW_h
            if self.b.requires_grad and self.b.grad is not None: self.b.grad += db

        out_full_sequence._backward = _backward
        
        # --- KRİTİK DEĞİŞİKLİK: Sadece son zaman adımını döndür ---
        # Bu, LSTM'den sonra gelen Linear katmanın doğru boyutta (N, H) girdi almasını sağlar.
        return out_full_sequence[:, -1, :]


class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.input_shape = None
    def forward(self, x: Tensor) -> Tensor:
        self.input_shape = x.data.shape
        N = self.input_shape[0]
        out = Tensor(x.data.reshape(N, -1), _children=(x,), _op="flatten", requires_grad=x.requires_grad)
        
        def _backward():
            if x.requires_grad and x.grad is not None:
                x.grad += out.grad.reshape(self.input_shape)
        
        out._backward = _backward
        return out

class Conv2D(Layer):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        limit = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weights = Tensor(xp.random.randn(out_channels, in_channels, kernel_size, kernel_size) * limit, requires_grad=True)
        self.bias = Tensor(xp.zeros(out_channels), requires_grad=True)
    def forward(self, x: Tensor) -> Tensor:
        return x.conv2d(self.weights, self.bias, self.stride, self.padding)
    def parameters(self) -> List[Tensor]:
        return [self.weights, self.bias]

class MaxPool2D(Layer):
    def __init__(self, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
    def forward(self, x: Tensor) -> Tensor:
        return x.max_pool2d(self.kernel_size, self.stride)
    def parameters(self) -> List[Tensor]:
        return []

class Embedding(Layer):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weights = Tensor(
            xp.random.randn(num_embeddings, embedding_dim) * 0.01,
            requires_grad=True
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.data.dtype not in [xp.int32, xp.int64]:
            x_int = xp.asarray(x.data, dtype=xp.int32)
        else:
            x_int = x.data
        return self.weights[x_int]

    def parameters(self) -> List[Tensor]:
        return [self.weights]


class Attention(Layer):
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.key_proj = Linear(input_dim, embed_dim)
        self.query_proj = Linear(input_dim, embed_dim)
        self.value_proj = Linear(input_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)
        
        k_t = k.transpose(0, 2, 1)
        scores = q.dot(k_t)
        
        scaled_scores = scores * (self.embed_dim ** -0.5)
        attention_weights = scaled_scores.softmax(axis=-1)
        
        context = attention_weights.dot(v)
        return context

    def parameters(self) -> List[Tensor]:
        return self.key_proj.parameters() + self.query_proj.parameters() + self.value_proj.parameters()