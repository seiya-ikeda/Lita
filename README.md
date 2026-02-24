# Lita (LLm with Innner Thoughts Action)

Inner Thoughtsフレームワークを実装した、自発的に話しかけてくれるAI友達Discord Bot

## 📖 概要

従来のチャットボットは「聞かれたら答える」反応型（Reactive）ですが、このBotは「自分から話しかける」能動型（Proactive）の会話AIです。

### 主な特徴

- **Inner Thoughts**: AIが内部で「思考」を生成し、適切なタイミングで発言
- **動機づけ評価**: 発言すべきかどうかを5段階でスコアリング
- **長期記憶**: ユーザーについての情報を記憶し、会話に活用
- **研究用ログ**: 全データを収集し、後で分析可能

## 🏗️ アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                     Discord Bot                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐     ┌──────────────────┐                 │
│  │ User Message │────▶│ Reactive Response │                 │
│  └──────────────┘     └──────────────────┘                 │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Inner Thoughts Engine                    │   │
│  │  ┌─────────┐  ┌───────────┐  ┌────────────┐         │   │
│  │  │ Trigger │─▶│  Thought  │─▶│ Motivation │         │   │
│  │  │         │  │ Formation │  │ Evaluation │         │   │
│  │  └─────────┘  └───────────┘  └─────┬──────┘         │   │
│  │                                     │                │   │
│  │                          Score > Threshold?          │   │
│  │                                     │                │   │
│  │                          ┌─────────┴─────────┐      │   │
│  │                          ▼                   ▼      │   │
│  │                    ┌──────────┐       ┌──────────┐  │   │
│  │                    │  Speak   │       │  Reserve │  │   │
│  │                    └──────────┘       └──────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────┐    ┌──────────────────┐              │
│  │  Memory Manager  │    │  Research Logger │              │
│  │  - Short-term    │    │  - Conversations │              │
│  │  - Long-term     │    │  - Thoughts      │              │
│  │  - Thought Pool  │    │  - Metrics       │              │
│  └──────────────────┘    └──────────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 セットアップ

### 1. 必要なもの

- Python 3.10+
- Discord Bot Token
- OpenAI API Key

### 2. インストール

```bash
# リポジトリをクローン（または自分のプロジェクトにファイルをコピー）
cd proactive-ai-friend

# 仮想環境を作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存関係をインストール
pip install -r requirements.txt
```

### 3. 環境変数の設定

```bash
# .envファイルを作成
cp .env.example .env

# .envを編集してAPIキーを設定
```

### 4. Discord Botの作成

1. [Discord Developer Portal](https://discord.com/developers/applications) にアクセス
2. "New Application" をクリック
3. "Bot" タブで "Add Bot"
4. "MESSAGE CONTENT INTENT" を有効化
5. Token をコピーして `.env` に設定
6. OAuth2 > URL Generator で `bot` スコープと必要な権限を選択
7. 生成されたURLでBotをサーバーに招待

### 5. 起動

```bash
python discord_bot.py
```

## 📝 使い方

### 基本的な会話

- **DM**: Botに直接DMを送る
- **サーバー**: `@Bot名 メッセージ` でメンション

### コマンド

| コマンド | 説明 |
|---------|------|
| `!status` | 現在のステータスと統計を表示 |
| `!memories` | Botが覚えていることを表示 |
| `!thoughts` | 保留中の思考を表示（デバッグ用） |
| `!forget` | 記憶をリセット |
| `!config` | 現在の設定を表示 |
| `!export` | ログをエクスポート |
| `!interests` | 記憶から抽出したユーザーの興味を表示 |
| `!serach` | ユーザーの興味のありそうなサイトを検索する (デバッグ用) |

## 🔬 研究用設定

### 実験条件の切り替え

`config.py` で実験条件を変更:

```python
# Proactive（能動型）モード
EXPERIMENT_CONDITION = "proactive"

# Reactive（反応型）モード - 比較用
EXPERIMENT_CONDITION = "reactive"
```

### 調整可能なパラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `MOTIVATION_THRESHOLD` | 3.5 | 発言の閾値（1-5） |
| `THOUGHT_GENERATION_INTERVAL` | 30秒 | 思考生成の間隔 |
| `SILENCE_TIMEOUT` | 300秒 | 沈黙後に話しかけるまでの時間 |
| `MAX_CONSECUTIVE_INTERVENTIONS` | 2 | 連続発言の最大回数 |

### 収集されるデータ

```
research_logs/
├── conversations/     # 全会話ログ（CSV）
├── thoughts/         # 思考生成・評価ログ（CSV）
└── metrics/          # セッション統計（CSV, JSON）
```

### 評価指標

1. **介入の適切さ**: 動機づけスコアの分布
2. **介入受容率**: 介入後にユーザーが返答した割合
3. **会話継続率**: セッション長、ターン数
4. **思考の発現率**: 生成された思考のうち発言された割合

## 📊 研究への発展

### 論文化に向けて

1. **比較実験**: Proactive vs Reactive条件での比較
2. **ユーザー調査**: アンケートによる主観評価
3. **パラメータ探索**: 閾値の最適化
4. **質的分析**: 会話ログの内容分析

### 差別化のアイデア

- 日本語での評価（ほぼ未開拓）
- 1対1の長期的関係性構築
- 感情認識との組み合わせ
- 「ちょうど良い」プロアクティブさの探求

## 🔧 カスタマイズ

### ペルソナの変更

`config.py` の `AI_PERSONA` を編集:

```python
AI_NAME = "あなたのAIの名前"
AI_PERSONA = """
あなたのAIのペルソナ設定...
"""
```

### 動機づけ基準の調整

`config.py` の `MOTIVATION_CRITERIA` を編集して、
発言判断の基準をカスタマイズできます。

## 📚 参考文献

- Liu, X. B., et al. (2025). "Proactive Conversational Agents with Inner Thoughts." CHI 2025.
- [Inner Thoughts Project Page](https://liubruce.me/inner_thoughts/)

## 📄 ライセンス

MIT License

---

**Note**: このプロジェクトは研究・教育目的で作成されています。
