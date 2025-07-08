from typing import List
from azuraforge_core import Tensor, xp
from .base import Layer
from .linear import Linear

class Embedding(Layer):
    """Girdi olarak tamsayı indeksleri alan ve her birini yoğun bir vektöre eşleyen katman."""
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Ağırlıkları küçük rastgele değerlerle başlat
        self.weights = Tensor(
            xp.random.randn(num_embeddings, embedding_dim) * 0.01,
            requires_grad=True
        )

    def forward(self, x: Tensor) -> Tensor:
        # Girdinin tamsayı olduğundan emin ol
        if x.data.dtype not in [xp.int32, xp.int64]:
            x_int = xp.asarray(x.data, dtype=xp.int32)
        else:
            x_int = x.data
        
        # İleri indeksleme ile embedding vektörlerini al
        return self.weights[x_int]

    def parameters(self) -> List[Tensor]:
        return [self.weights]

class Attention(Layer):
    """Basit bir Self-Attention (Dot-Product) katmanı."""
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
        
        k_t = k.transpose(0, 2, 1) # (N, embed_dim, T)
        
        # Dikkat skorlarını hesapla
        scores = q.dot(k_t)
        
        # Ölçekleme
        scaled_scores = scores * (self.embed_dim ** -0.5)
        
        # Softmax ile ağırlıkları normalize et
        attention_weights = scaled_scores.softmax(axis=-1)
        
        # Değer (value) vektörleri ile ağırlıklı toplamı al
        context = attention_weights.dot(v)
        return context

    def parameters(self) -> List[Tensor]:
        return self.key_proj.parameters() + self.query_proj.parameters() + self.value_proj.parameters()

