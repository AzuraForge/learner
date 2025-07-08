from typing import List, Tuple, Optional
import numpy as np
from azuraforge_core import Tensor, xp
from .base import Layer

class LSTM(Layer):
    """
    Long Short-Term Memory (LSTM) katmanı. Zaman serisi gibi sıralı veriler için tasarlanmıştır.
    """
    def __init__(self, input_size: int, hidden_size: int, return_sequences: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences
        
        H = hidden_size
        D = input_size
        
        # Xavier/Glorot başlatması
        limit = np.sqrt(6.0 / (D + H))
        
        # Giriş, unutma, hücre ve çıkış kapıları için ağırlıkları tek bir matriste birleştiriyoruz.
        self.W_x = Tensor(xp.random.uniform(-limit, limit, (D, H * 4)), requires_grad=True)
        self.W_h = Tensor(xp.random.uniform(-limit, limit, (H, H * 4)), requires_grad=True)
        self.b = Tensor(xp.zeros(H * 4), requires_grad=True)
        
        self.cache: Optional[Tuple] = None

    def parameters(self) -> List[Tensor]:
        return [self.W_x, self.W_h, self.b]

    def forward(self, x: Tensor) -> Tensor:
        if x.data.ndim == 2:
            # Girdiyi (N, T) -> (N, T, 1) şeklinde yeniden boyutlandır
            x = Tensor(x.data.reshape(*x.data.shape, 1), _children=(x,), requires_grad=x.requires_grad)

        N, T, D = x.data.shape
        H = self.hidden_size

        h_prev = xp.zeros((N, H), dtype=xp.float32)
        c_prev = xp.zeros((N, H), dtype=xp.float32)
        
        h_all = xp.zeros((N, T, H), dtype=xp.float32)
        c_all = xp.zeros((N, T, H), dtype=xp.float32)
        i_all = xp.zeros((N, T, H), dtype=xp.float32)
        f_all = xp.zeros((N, T, H), dtype=xp.float32)
        o_all = xp.zeros((N, T, H), dtype=xp.float32)
        g_all = xp.zeros((N, T, H), dtype=xp.float32)

        for t in range(T):
            x_t = x.data[:, t, :]
            gates = x_t @ self.W_x.data + h_prev @ self.W_h.data + self.b.data
            
            # Kapıları ayır (input, forget, output, gate)
            i = 1 / (1 + xp.exp(-gates[:, :H]))
            f = 1 / (1 + xp.exp(-gates[:, H:2*H]))
            o = 1 / (1 + xp.exp(-gates[:, 2*H:3*H]))
            g = xp.tanh(gates[:, 3*H:])
            
            c_next = f * c_prev + i * g
            h_next = o * xp.tanh(c_next)

            h_prev, c_prev = h_next, c_next
            
            h_all[:, t, :] = h_next
            c_all[:, t, :] = c_next
            i_all[:, t, :] = i
            f_all[:, t, :] = f
            o_all[:, t, :] = o
            g_all[:, t, :] = g

        out_full_sequence = Tensor(h_all, _children=(x, self.W_x, self.W_h, self.b), _op="lstm", requires_grad=x.requires_grad)
        
        self.cache = (x.data, h_all, c_all, i_all, f_all, o_all, g_all)

        def _backward():
            if not out_full_sequence.requires_grad or out_full_sequence.grad is None: return
            assert self.cache is not None, "Cache is not set for LSTM backward pass"
            
            x_data, h_data, c_data, i_data, f_data, o_data, g_data = self.cache
            _N, _T, _ = x_data.shape
            _H = self.hidden_size
            
            dx = xp.zeros_like(x_data)
            dW_x = xp.zeros_like(self.W_x.data)
            dW_h = xp.zeros_like(self.W_h.data)
            db = xp.zeros_like(self.b.data)
            
            dh_next = xp.zeros((_N, _H), dtype=xp.float32)
            dc_next = xp.zeros((_N, _H), dtype=xp.float32)

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
                h_prev_t = h_data[:, t-1, :] if t > 0 else xp.zeros((_N, _H), dtype=xp.float32)
                
                if x.requires_grad: dx[:, t, :] = dgates @ self.W_x.data.T
                dh_next = dgates @ self.W_h.data.T
                dc_next = dc * f_data[:, t, :]
                
                dW_x += x_t.T @ dgates
                dW_h += h_prev_t.T @ dgates
                db += xp.sum(dgates, axis=0)

            if x.requires_grad and x.grad is not None: x.grad += dx
            if self.W_x.requires_grad and self.W_x.grad is not None: self.W_x.grad += dW_x
            if self.W_h.requires_grad and self.W_h.grad is not None: self.W_h.grad += dW_h
            if self.b.requires_grad and self.b.grad is not None: self.b.grad += db

        out_full_sequence._backward = _backward
        
        if self.return_sequences:
            return out_full_sequence
        else:
            return out_full_sequence[:, -1, :]
