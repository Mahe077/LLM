import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Play, Pause, AlertTriangle, BookOpen, Code } from 'lucide-react';

let trainingInterval = null;

const CryptoTransformerTutorial = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [isTraining, setIsTraining] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState(1.0);
  const [cryptoData, setCryptoData] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [selectedTopic, setSelectedTopic] = useState('concepts');
  const [showMath, setShowMath] = useState(false);

  useEffect(() => {
    const generateData = () => {
      const data = [];
      let price = 50000;
      for (let i = 0; i < 100; i++) {
        price += (Math.random() - 0.5) * 2000 + Math.sin(i * 0.1) * 500;
        data.push({
          time: i,
          price: Math.max(price, 1000),
          volatility: Math.random() * 0.05 + 0.01,
          volume: Math.random() * 1000000
        });
      }
      return data;
    };
    setCryptoData(generateData());

    return () => {
      if (trainingInterval) {
        clearInterval(trainingInterval);
      }
    };
  }, []);

  const topics = {
    concepts: {
      title: "Core Concepts Deep Dive",
      steps: [
        {
          title: "1. The Attention Revolution",
          description: "The paper states: 'Attention mechanisms have become an integral part of compelling sequence modeling.' But WHY is this revolutionary for crypto trading?",
          concept: "Mathematical Foundation",
          deepDive: `
**The Problem with RNNs in Trading:**
RNNs process h_t = f(h_{t-1}, x_t), meaning they must pass information sequentially. 
For a 60-day trading sequence, information from day 1 must pass through 59 transformations 
to affect the final prediction. This creates the vanishing gradient problem.

**Transformer Solution:**
All positions can directly attend to each other in O(1) operations. 
If Bitcoin crashed 30 days ago due to regulatory news, the model can directly 
connect that event to current market conditions without information decay.

**Why This Matters for Crypto:**
Crypto markets are highly event-driven. A single tweet, regulatory announcement, 
or whale movement can instantly change market dynamics. Traditional RNNs struggle 
to maintain these long-range dependencies effectively.`,
          code: `
# RNN Information Flow (Sequential)
h_1 = f(h_0, price_1)    # Day 1
h_2 = f(h_1, price_2)    # Day 2, depends on h_1
...
h_60 = f(h_59, price_60) # Day 60, info from day 1 is heavily degraded

# Transformer Attention (Direct)
attention_weights = softmax(Q @ K^T / sqrt(d_k))
# Every day can directly attend to every other day
# Day 60 can have strong attention to Day 1 if needed

# Practical Example for Bitcoin
if regulatory_news_day_1 and current_volatility > threshold:
  attention_weight[60][1] = high_value  # Direct connection!
        `
        },
        {
          title: "2. Positional Encoding Mathematics",
          description: "The sinusoidal encoding PE(pos,2i) = sin(pos/10000^(2i/d_model)) creates unique representations for each time step.",
          concept: "Time Embedding Theory",
          deepDive: `
**Mathematical Intuition:**
Each dimension uses a different frequency. Low frequencies (sin, cos with large periods) 
capture long-term patterns, high frequencies capture short-term patterns.

**Why Sine/Cosine?**
1. Bounded values (-1, 1) prevent gradient explosion
2. Relative position can be computed: PE(pos+k) = linear_function(PE(pos))
3. Different frequencies create orthogonal representations

**Crypto Trading Application:**
- Low frequencies (d=0,1): Weekly/monthly trends
- Medium frequencies (d=32-64): Daily cycles, weekend effects  
- High frequencies (d=500+): Intraday patterns, market open/close

**The Paper's Insight:**
"We chose this function because it would allow the model to easily learn 
to attend by relative positions." This is crucial for trading - we care about 
"2 days after earnings" not just "day 47 of our sequence."`,
          code: `
import numpy as np
import matplotlib.pyplot as plt

def positional_encoding_crypto(seq_length, d_model):
  """
  Create positional encodings for crypto time series
  seq_length: number of time periods (hours/days)
  d_model: embedding dimension
  """
  pos_enc = np.zeros((seq_length, d_model))
  
  for pos in range(seq_length):
      for i in range(0, d_model, 2):
          # Different frequencies for different patterns
          freq = 1.0 / (10000 ** ((2 * i) / d_model))
          
          pos_enc[pos, i] = np.sin(pos * freq)      # Sine component
          pos_enc[pos, i + 1] = np.cos(pos * freq)  # Cosine component
  
  return pos_enc

# Visualize what this looks like
pos_enc = positional_encoding_crypto(100, 64)
print(f"Position encoding shape: {pos_enc.shape}")
print(f"Each time step has unique {d_model}-dim representation")

# Key insight: Similar time periods have similar encodings
similarity_1_2 = np.dot(pos_enc[1], pos_enc[2])    # High similarity
similarity_1_50 = np.dot(pos_enc[1], pos_enc[50])  # Lower similarity
        `
        },
        {
          title: "3. Multi-Head Attention Deep Dive",
          description: "The paper uses 8 heads with d_k = d_v = 64. Each head learns different relationships. For crypto, this is powerful pattern recognition.",
          concept: "Parallel Pattern Detection",
          deepDive: `
**The Mathematical Framework:**
MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW_Q^i, KW_K^i, VW_V^i)

Each head has its own learned projection matrices, so they can specialize 
in different types of patterns.

**Crypto-Specific Attention Patterns:**
Head 1: Price momentum patterns (recent trends)
Head 2: Volume-price divergences  
Head 3: Volatility clustering (high vol periods)
Head 4: Support/resistance levels
Head 5: Cross-asset correlations (BTC vs ETH)
Head 6: Time-of-day effects
Head 7: News/event impact windows
Head 8: Mean reversion signals

**Why 8 Heads Work Well:**
The paper found that 16 heads hurt performance - too much redundancy.
1 head was 0.9 BLEU worse - not enough specialization.
8 heads hit the sweet spot for most sequence modeling tasks.

**Attention Weight Interpretation:**
High attention weights show which historical periods the model considers 
most relevant for current predictions. This creates interpretable trading signals!`,
          code: `
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CryptoMultiHeadAttention(nn.Module):
  def __init__(self, d_model=256, num_heads=8):
      super().__init__()
      assert d_model % num_heads == 0
      
      self.d_model = d_model
      self.num_heads = num_heads  
      self.d_k = d_model // num_heads  # 256 // 8 = 32
      
      # Linear projections for Q, K, V
      self.W_q = nn.Linear(d_model, d_model)
      self.W_k = nn.Linear(d_model, d_model) 
      self.W_v = nn.Linear(d_model, d_model)
      self.W_o = nn.Linear(d_model, d_model)
      
  def forward(self, query, key, value, mask=None):
      batch_size, seq_len = query.size(0), query.size(1)
      
      # 1. Linear projections and reshape for multi-head
      Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)
      K = self.W_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2) 
      V = self.W_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)
      
      # 2. Scaled dot-product attention for each head
      attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
      
      # 3. Concatenate heads and apply output projection
      attention_output = attention_output.transpose(1,2).contiguous().view(
          batch_size, seq_len, self.d_model)
      
      return self.W_o(attention_output), attention_weights
  
  def scaled_dot_product_attention(self, Q, K, V, mask=None):
      scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
      
      if mask is not None:
          scores = scores.masked_fill(mask == 0, -1e9)
          
      attention_weights = F.softmax(scores, dim=-1)
      attention_output = torch.matmul(attention_weights, V)
      
      return attention_output, attention_weights

# Usage for crypto sequences
# Input: [batch_size, seq_length, d_model] 
# seq_length = 60 (60 days of price history)
# d_model = 256 (embedding dimension)
crypto_attention = CryptoMultiHeadAttention(d_model=256, num_heads=8)
        `
        }
      ]
    },
    implementation: {
      title: "Implementation Details",
      steps: [
        {
          title: "4. Data Pipeline Architecture", 
          description: "Real-world crypto data is messy, has gaps, and comes from multiple sources. Here's how to build a robust pipeline.",
          concept: "Production Data Handling",
          deepDive: `
**Key Challenges:**
1. **Missing Data**: Exchanges go down, APIs fail, data gaps occur
2. **Multiple Timeframes**: Need 1min, 5min, 1hour, 1day simultaneously  
3. **Multiple Assets**: BTC, ETH, ADA, etc. with different liquidity
4. **Feature Engineering**: Raw price isn't enough - need technical indicators
5. **Real-time vs Batch**: Training on historical, predicting on live data

**Data Schema Design:**
Instead of simple [price, volume], we need rich feature vectors:
- OHLCV (Open, High, Low, Close, Volume)
- Technical indicators (RSI, MACD, Bollinger Bands)
- Market microstructure (bid-ask spread, order book depth)
- Cross-asset features (BTC dominance, DeFi TVL)
- Macro features (VIX, DXY, etc.)

**The Transformer Advantage:**
Unlike RNNs that need fixed-length sequences, Transformers can handle 
variable-length inputs with attention masks. This lets us handle:
- Weekend gaps in traditional markets
- Exchange downtime periods  
- Different listing dates for new tokens`,
          code: `
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import ccxt  # Crypto exchange library
from dataclasses import dataclass

@dataclass
class CryptoFeatures:
  """Rich feature representation for crypto data"""
  timestamp: int
  symbol: str
  
  # OHLCV
  open: float
  high: float  
  low: float
  close: float
  volume: float
  
  # Technical indicators
  rsi: float
  macd: float
  bb_upper: float
  bb_lower: float
  sma_20: float
  ema_50: float
  
  # Market microstructure
  bid_ask_spread: float
  market_cap: float
  
  # Cross-asset features
  btc_dominance: float
  eth_ratio: float  # ETH/BTC ratio
  
  # Sentiment/macro
  fear_greed_index: float
  funding_rate: float

class CryptoDataPipeline:
  def __init__(self, symbols: List[str], timeframe: str = '1h'):
      self.symbols = symbols
      self.timeframe = timeframe
      self.exchanges = {
          'binance': ccxt.binance(),
          'coinbase': ccxt.coinbasepro(),
          # Add redundancy for reliability
      }
      
  def fetch_ohlcv(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
      """Fetch OHLCV with redundancy across exchanges"""
      for exchange_name, exchange in self.exchanges.items():
          try:
              ohlcv = exchange.fetch_ohlcv(symbol, self.timeframe, limit=limit)
              df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
              df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
              return df
          except Exception as e:
              print(f"Failed to fetch from {exchange_name}: {e}")
              continue
      raise Exception(f"Failed to fetch {symbol} from all exchanges")

  def calculate_rsi(self, series, period=14):
      delta = series.diff(1)
      gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
      loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
      rs = gain / loss
      return 100 - (100 / (1 + rs))
  
  def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
      """Add technical indicators using pandas_ta or custom functions"""
      # RSI (Relative Strength Index)
      df['rsi'] = self.calculate_rsi(df['close'])
      
      # Moving averages
      df['sma_20'] = df['close'].rolling(20).mean()
      df['ema_50'] = df['close'].ewm(span=50).mean()
      
      # Bollinger Bands
      rolling_mean = df['close'].rolling(20).mean()
      rolling_std = df['close'].rolling(20).std()
      df['bb_upper'] = rolling_mean + (rolling_std * 2)
      df['bb_lower'] = rolling_mean - (rolling_std * 2)
      
      # MACD
      exp1 = df['close'].ewm(span=12).mean()
      exp2 = df['close'].ewm(span=26).mean()
      df['macd'] = exp1 - exp2
      
      return df
  
  def create_sequences(self, df: pd.DataFrame, seq_length: int = 60, 
                      predict_horizon: int = 1) -> Tuple[np.array, np.array]:
      """Create sequences for Transformer training"""
      # Normalize features
      feature_cols = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 
                      'sma_20', 'ema_50', 'bb_upper', 'bb_lower']
      
      # Use log returns instead of raw prices for better stationarity
      for col in ['open', 'high', 'low', 'close']:
          df[f'{col}_return'] = np.log(df[col] / df[col].shift(1))
      
      # Prepare sequences
      sequences = []
      targets = []
      
      for i in range(seq_length, len(df) - predict_horizon):
          # Input sequence: past seq_length periods
          seq = df[feature_cols].iloc[i-seq_length:i].values
          
          # Target: future return
          current_price = df['close'].iloc[i]  
          future_price = df['close'].iloc[i + predict_horizon]
          target = (future_price - current_price) / current_price
          
          sequences.append(seq)
          targets.append(target)
      
      return np.array(sequences), np.array(targets)

# Usage example
pipeline = CryptoDataPipeline(['BTC/USDT', 'ETH/USDT'])
btc_data = pipeline.fetch_ohlcv('BTC/USDT')
btc_data = pipeline.add_technical_indicators(btc_data)
sequences, targets = pipeline.create_sequences(btc_data)

print(f"Created {len(sequences)} sequences of shape {sequences[0].shape}")
print(f"Target shape: {targets.shape}")
        `
        },
        {
          title: "5. Transformer Architecture for Finance",
          description: "The standard Transformer needs modifications for financial time series. Here's a complete implementation with financial-specific improvements.",
          concept: "Finance-Optimized Architecture", 
          deepDive: `
**Key Modifications from Standard Transformer:**

1. **Causal Attention**: We can't look into the future (no lookahead bias)
2. **Price-Aware Embeddings**: Handle different price scales across assets  
3. **Multi-Task Heads**: Predict price, volatility, and direction simultaneously
4. **Temporal Convolutions**: Add local pattern recognition before attention
5. **Regularization**: Heavy dropout and layer norm for noisy financial data

**Architecture Decisions:**
- **Model Size**: 6 layers, 256 dim, 8 heads (smaller than NLP models)
- Financial data has lower complexity than language
- Prevents overfitting on limited training data
- **Sequence Length**: 60-120 periods 
- Long enough for patterns, short enough for stable gradients
- **Multi-Scale**: Process 1hour, 4hour, daily simultaneously

**Loss Function Design:**
Standard MSE isn't optimal for trading. We use:
1. Directional accuracy loss (sign prediction)
2. Magnitude loss (how much price moves)  
3. Risk-adjusted loss (penalize high-volatility predictions)
4. Transaction cost penalty`,
          code: `
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict

class FinancialTransformer(nn.Module):
  def __init__(self, 
                input_dim: int = 20,      # Number of features 
                d_model: int = 256,       # Model dimension
                num_heads: int = 8,       # Attention heads
                num_layers: int = 6,      # Transformer layers
                seq_length: int = 60,     # Input sequence length
                dropout: float = 0.1):
      super().__init__()
      
      self.d_model = d_model
      self.seq_length = seq_length
      
      # 1. Input embedding with price scaling
      self.input_projection = nn.Linear(input_dim, d_model)
      self.price_embedding = PriceAwareEmbedding(d_model)
      
      # 2. Positional encoding (learnable for financial data)
      self.pos_encoding = LearnablePositionalEncoding(seq_length, d_model)
      
      # 3. Temporal convolution for local patterns
      self.temporal_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
      
      # 4. Transformer encoder layers with causal masking
      encoder_layer = nn.TransformerEncoderLayer(
          d_model=d_model,
          nhead=num_heads, 
          dim_feedforward=d_model * 4,
          dropout=dropout,
          activation='gelu',  # Better than ReLU for financial data
          batch_first=True
      )
      self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
      
      # 5. Multi-task output heads
      self.price_head = nn.Linear(d_model, 1)      # Price prediction
      self.volatility_head = nn.Linear(d_model, 1)  # Volatility prediction  
      self.direction_head = nn.Linear(d_model, 3)   # Up/Down/Sideways
      
      # 6. Output normalization
      self.output_norm = nn.LayerNorm(d_model)
      self.dropout = nn.Dropout(dropout)
      
  def create_causal_mask(self, seq_len: int) -> torch.Tensor:
      """Create causal mask to prevent looking into future"""
      mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
      return mask.bool()
  
  def forward(self, x: torch.Tensor, prices: torch.Tensor) -> Dict[str, torch.Tensor]:
      """
      Args:
          x: [batch_size, seq_length, input_dim] - feature sequences
          prices: [batch_size, seq_length, 1] - price sequences for embedding
      Returns:
          Dictionary with predictions for price, volatility, direction
      """
      batch_size, seq_len, _ = x.shape
      
      # 1. Input embedding and projection
      x = self.input_projection(x)  # [batch, seq, d_model]
      
      # 2. Add price-aware embedding
      price_emb = self.price_embedding(prices)
      x = x + price_emb
      
      # 3. Add positional encoding
      x = x + self.pos_encoding(x)
      x = self.dropout(x)
      
      # 4. Temporal convolution for local patterns
      x_conv = self.temporal_conv(x.transpose(1, 2)).transpose(1, 2)
      x = x + x_conv  # Residual connection
      
      # 5. Create causal mask (can't look into future)
      causal_mask = self.create_causal_mask(seq_len).to(x.device)
      
      # 6. Transformer processing
      x = self.transformer(x, mask=causal_mask)
      x = self.output_norm(x)
      
      # 7. Use only the last time step for prediction
      last_hidden = x[:, -1, :]  # [batch, d_model]
      
      # 8. Multi-task predictions
      price_pred = self.price_head(last_hidden)
      volatility_pred = torch.exp(self.volatility_head(last_hidden))  # Always positive
      direction_logits = self.direction_head(last_hidden)
      
      return {
          'price': price_pred,
          'volatility': volatility_pred,
          'direction': F.softmax(direction_logits, dim=-1),
          'hidden': last_hidden
      }

class PriceAwareEmbedding(nn.Module):
  """Embedding that handles different price scales across assets"""
  def __init__(self, d_model: int):
      super().__init__()
      self.d_model = d_model
      self.price_proj = nn.Linear(1, d_model)
      
  def forward(self, prices: torch.Tensor) -> torch.Tensor:
      # Log-normalize prices to handle different scales
      log_prices = torch.log(prices + 1e-8)  # Avoid log(0)
      return self.price_proj(log_prices)

class LearnablePositionalEncoding(nn.Module):
  """Learnable positional encoding for financial sequences"""
  def __init__(self, seq_length: int, d_model: int):
      super().__init__()
      self.pos_embedding = nn.Parameter(torch.randn(seq_length, d_model))
      
  def forward(self, x: torch.Tensor) -> torch.Tensor:
      seq_len = x.size(1)
      return self.pos_embedding[:seq_len, :].unsqueeze(0).expand(x.size(0), -1, -1)

# Multi-task loss function
class FinancialLoss(nn.Module):
  def __init__(self, price_weight: float = 1.0, vol_weight: float = 0.5, 
                direction_weight: float = 0.3):
      super().__init__()
      self.price_weight = price_weight
      self.vol_weight = vol_weight  
      self.direction_weight = direction_weight
      
  def forward(self, predictions: Dict[str, torch.Tensor], 
              targets: Dict[str, torch.Tensor]) -> torch.Tensor:
      
      # Price prediction loss (MSE)
      price_loss = F.mse_loss(predictions['price'], targets['price'])
      
      # Volatility prediction loss (MSE)
      vol_loss = F.mse_loss(predictions['volatility'], targets['volatility'])
      
      # Directional prediction loss (Cross-Entropy)
      direction_loss = F.cross_entropy(predictions['direction'], targets['direction'])
      
      total_loss = (self.price_weight * price_loss + 
                    self.vol_weight * vol_loss + 
                    self.direction_weight * direction_loss)
                    
      return total_loss
        `
        }
      ]
    }
  };

  const simulateTraining = () => {
    if (isTraining) {
      clearInterval(trainingInterval);
      setIsTraining(false);
    } else {
      setIsTraining(true);
      setEpoch(0);
      setLoss(1.0);
      setPredictions([]);
      
      trainingInterval = setInterval(() => {
        setEpoch(prev => {
          const newEpoch = prev + 1;
          
          if (newEpoch >= 100) {
            clearInterval(trainingInterval);
            setIsTraining(false);
            const preds = cryptoData.slice(-30).map((point, i) => ({
              time: point.time + i + 1,
              actual: point.price,
              predicted: point.price * (1 + (Math.random() - 0.5) * 0.015),
              confidence_upper: point.price * (1 + Math.abs(Math.random() - 0.5) * 0.025),
              confidence_lower: point.price * (1 - Math.abs(Math.random() - 0.5) * 0.025),
              volatility_pred: point.volatility * (0.8 + Math.random() * 0.4)
            }));
            setPredictions(preds);
            return newEpoch;
          }
          
          const newLoss = Math.max(0.05, 1.0 * Math.exp(-newEpoch * 0.03));
          setLoss(newLoss);
          
          return newEpoch;
        });
      }, 80);
    }
  };

  const renderAttentionHeatmap = () => {
    if (predictions.length > 0) {
      return (
        <div className="mt-4 p-4 bg-gray-50 rounded-lg">
          <h3 className="font-medium text-gray-800 mb-2">Attention Heatmap (Placeholder)</h3>
          <div className="text-sm text-gray-600">
            This would visualize which past time steps the model is paying attention to for its prediction.
          </div>
        </div>
      );
    }
    return null;
  };

  const renderVolatilityChart = () => {
    if (cryptoData.length > 0) {
        const volData = cryptoData.slice(-60).map(d => ({ time: d.time, volatility: d.volatility }));
        if (predictions.length > 0) {
            const predictionMap = new Map(predictions.map(p => [p.time, p.volatility_pred]));
            volData.forEach(d => {
                if (predictionMap.has(d.time)) {
                    d.predicted_volatility = predictionMap.get(d.time);
                }
            });
        }

      return (
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={volData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis domain={[0, 0.1]} />
            <Tooltip />
            <Line type="monotone" dataKey="volatility" stroke="#f97316" strokeWidth={2} name="Historical Vol" dot={false} />
            {predictions.length > 0 && (
                <Line type="monotone" dataKey="predicted_volatility" stroke="#a855f7" strokeWidth={2} name="Predicted Vol" strokeDasharray="5 5" dot={false} />
            )}
          </LineChart>
        </ResponsiveContainer>
      );
    }
    return null;
  };

  const currentTopic = topics[selectedTopic];
  if (!currentTopic) return null;
  
  const currentStepData = currentTopic.steps[currentStep] || currentTopic.steps[0];
  if (!currentStepData) return null;

  return (
    <div className="max-w-7xl mx-auto p-6 bg-gray-50 min-h-screen">
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <div className="flex items-center gap-3 mb-4">
          <BookOpen className="text-blue-600" size={24} />
          <h1 className="text-2xl font-bold text-gray-800">
            Transformer for Crypto Trading - Comprehensive Learning Guide
          </h1>
        </div>
        
        <div className="bg-red-50 border-l-4 border-red-400 p-4 mb-4">
          <div className="flex items-start">
            <AlertTriangle className="text-red-400 mt-1 mr-2" size={16} />
            <div>
              <h3 className="font-semibold text-red-800">Critical Risk Warning</h3>
              <p className="text-red-700">
                Cryptocurrency trading involves substantial risk of loss and is not suitable for everyone. 
                This is educational content only. Past performance does not guarantee future results.
                Never trade with money you cannot afford to lose entirely.
              </p>
            </div>
          </div>
        </div>

        <div className="flex gap-4 mb-4">
          {Object.entries(topics).map(([key, topic]) => (
            <button
              key={key}
              onClick={() => {setSelectedTopic(key); setCurrentStep(0);}}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                selectedTopic === key 
                  ? 'bg-blue-500 text-white' 
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              {topic.title}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold mb-4">{currentTopic.title}</h2>
          <div className="space-y-2">
            {currentTopic.steps.map((step, index) => (
              <div
                key={index}
                className={`p-3 rounded cursor-pointer transition-colors ${
                  index === currentStep 
                    ? 'bg-blue-100 border-blue-300 border' 
                    : 'bg-gray-50 hover:bg-gray-100'
                }`}
                onClick={() => setCurrentStep(index)}
              >
                <h3 className="font-medium text-sm">{step.title}</h3>
                <p className="text-xs text-gray-600">{step.concept}</p>
              </div>
            ))}
          </div>

          <div className="mt-6 p-4 bg-blue-50 rounded-lg">
            <h3 className="font-medium text-blue-800 mb-2">Learning Progress</h3>
            <div className="w-full bg-blue-200 rounded-full h-2">
              <div 
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${((currentStep + 1) / currentTopic.steps.length) * 100}%` }}
              />
            </div>
            <p className="text-xs text-blue-600 mt-1">
              Step {currentStep + 1} of {currentTopic.steps.length}
            </p>
          </div>
        </div>

        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold">Market Data & Predictions</h2>
              <div className="flex items-center gap-2">
                <button
                  onClick={simulateTraining}
                  className={`flex items-center gap-2 px-4 py-2 rounded ${
                    isTraining 
                      ? 'bg-red-500 text-white hover:bg-red-600' 
                      : 'bg-green-500 text-white hover:bg-green-600'
                  }`}
                >
                  {isTraining ? <Pause size={16} /> : <Play size={16} />}
                  {isTraining ? 'Stop Training' : 'Train Model'}
                </button>
                <button
                  onClick={() => setShowMath(!showMath)}
                  className="px-3 py-2 bg-purple-500 text-white rounded hover:bg-purple-600"
                >
                  {showMath ? 'Hide Math' : 'Show Math'}
                </button>
              </div>
            </div>
            
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={cryptoData.slice(-60)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis domain={['dataMin - 1000', 'dataMax + 1000']} />
                <Tooltip />
                <Line type="monotone" dataKey="price" stroke="#2563eb" strokeWidth={2} name="BTC Price" dot={false}/>
                {predictions.length > 0 && (
                  <Line 
                    type="monotone" 
                    dataKey="predicted" 
                    stroke="#ef4444" 
                    strokeWidth={2} 
                    strokeDasharray="5 5"
                    name="Predicted"
                    data={predictions}
                    dot={false}
                  />
                )}
              </LineChart>
            </ResponsiveContainer>

            {isTraining && (
              <div className="mt-4 p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg">
                <div className="flex justify-between items-center mb-2">
                  <span className="font-medium">Training Transformer Model</span>
                  <span className="text-sm text-gray-600">Epoch {epoch}/100</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3 mb-2">
                  <div 
                    className="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full transition-all duration-300"
                    style={{ width: `${epoch}%` }}
                  />
                </div>
                <div className="flex justify-between text-sm text-gray-600">
                  <span>Loss: {loss.toFixed(6)}</span>
                  <span>Learning Rate: {(0.001 * Math.exp(-epoch * 0.01)).toExponential(2)}</span>
                </div>
              </div>
            )}

            {renderAttentionHeatmap()}
          </div>

          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-semibold mb-3">Volatility & Risk Analysis</h3>
            {renderVolatilityChart()}
            
            {predictions.length > 0 && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
                <div className="p-3 bg-green-50 rounded">
                  <div className="text-xs text-green-600">Predicted Return</div>
                  <div className="text-lg font-bold text-green-800">
                    {((predictions[predictions.length-1].predicted / predictions[predictions.length-1].actual - 1) * 100).toFixed(2)}%
                  </div>
                </div>
                <div className="p-3 bg-blue-50 rounded">
                  <div className="text-xs text-blue-600">Confidence Interval</div>
                  <div className="text-lg font-bold text-blue-800">
                    ±{((predictions[predictions.length-1].confidence_upper / predictions[predictions.length-1].actual - 1) * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="p-3 bg-orange-50 rounded">
                  <div className="text-xs text-orange-600">Vol Forecast</div>
                  <div className="text-lg font-bold text-orange-800">
                    {(predictions[predictions.length-1].volatility_pred * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="p-3 bg-purple-50 rounded">
                  <div className="text-xs text-purple-600">Risk Score</div>
                  <div className="text-lg font-bold text-purple-800">
                    {Math.min(10, Math.max(1, Math.round(predictions[predictions.length-1].volatility_pred * 15)))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-lg p-6 mt-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">{currentStepData.title}</h2>
          <div className="text-sm text-gray-500">
            {currentStepData.concept}
          </div>
        </div>

        <div className="prose max-w-none mb-6">
          <div className="text-gray-700 leading-relaxed whitespace-pre-line" dangerouslySetInnerHTML={{ __html: currentStepData.description }} />
          
          {currentStepData.deepDive && (
            <div className="mt-4 p-4 bg-amber-50 border-l-4 border-amber-400 rounded">
              <h3 className="font-semibold text-amber-800 mb-2">Deep Dive</h3>
              <div className="text-amber-900 whitespace-pre-line text-sm" dangerouslySetInnerHTML={{ __html: currentStepData.deepDive }} />
            </div>
          )}
        </div>

        {showMath && currentStepData.concept.includes('Mathematical') && (
          <div className="mb-6 p-4 bg-purple-50 border border-purple-200 rounded-lg">
            <h3 className="font-semibold text-purple-800 mb-3">Mathematical Foundation</h3>
            <div className="space-y-3 text-sm text-purple-900">
              <div className="p-3 bg-white border rounded">
                <strong>Attention Formula:</strong><br/>
                Attention(Q,K,V) = softmax(QK^T / √d_k)V
              </div>
              <div className="p-3 bg-white border rounded">
                <strong>Positional Encoding:</strong><br/>
                PE(pos,2i) = sin(pos/10000^(2i/d_model))<br/>
                PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
              </div>
              <div className="p-3 bg-white border rounded">
                <strong>Multi-Head Output:</strong><br/>
                MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O<br/>
                where head_i = Attention(QW_Q^i, KW_K^i, VW_V^i)
              </div>
            </div>
          </div>
        )}

        <div className="bg-gray-900 text-green-400 p-6 rounded-lg font-mono text-sm overflow-x-auto">
          <div className="flex items-center gap-2 mb-4 text-white">
            <Code size={16} />
            <span className="font-semibold">Implementation Code</span>
          </div>
          <pre className="whitespace-pre-wrap">{currentStepData.code}</pre>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-lg p-6 mt-6">
        <h2 className="text-xl font-semibold mb-4">Next Steps for Implementation</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <div className="p-4 border border-blue-200 rounded-lg bg-blue-50">
            <h3 className="font-medium text-blue-800 mb-3">Immediate Actions (Week 1-2)</h3>
            <ul className="text-sm text-blue-700 space-y-2">
              <li>• Set up development environment (Python, PyTorch)</li>
              <li>• Download historical crypto data (1+ years)</li>
              <li>• Implement basic data pipeline</li>
              <li>• Create simple baseline model (moving average)</li>
              <li>• Set up paper trading account</li>
            </ul>
          </div>
          
          <div className="p-4 border border-green-200 rounded-lg bg-green-50">
            <h3 className="font-medium text-green-800 mb-3">Short Term (Month 1-2)</h3>
            <ul className="text-sm text-green-700 space-y-2">
              <li>• Implement Transformer architecture</li>
              <li>• Add technical indicators as features</li>
              <li>• Build robust backtesting framework</li>
              <li>• Test on multiple cryptocurrencies</li>
              <li>• Implement basic risk management</li>
            </ul>
          </div>
          
          <div className="p-4 border border-purple-200 rounded-lg bg-purple-50">
            <h3 className="font-medium text-purple-800 mb-3">Long Term (Month 3-6)</h3>
            <ul className="text-sm text-purple-700 space-y-2">
              <li>• Optimize hyperparameters systematically</li>
              <li>• Add alternative data sources</li>
              <li>• Implement portfolio management</li>
              <li>• Start with very small live positions</li>
              <li>• Monitor and iterate on strategy</li>
            </ul>
          </div>
        </div>
        
        <div className="mt-6 p-4 bg-gray-50 border border-gray-200 rounded-lg">
          <h3 className="font-medium text-gray-800 mb-2">Reality Check</h3>
          <p className="text-sm text-gray-600">
            Building a profitable trading system is extremely difficult. Most fail because they underestimate 
            transaction costs, overfit to historical data, or lack proper risk management. Focus on learning 
            the process rather than expecting immediate profits. Consider this a long-term educational project 
            that may or may not become profitable.
          </p>
        </div>
      </div>
    </div>
  );
};

export default CryptoTransformerTutorial;
