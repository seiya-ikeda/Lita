"""
Proactive AI Friend - Prompt Templates
思考生成、動機づけ評価、応答生成のプロンプト
"""

import config

# =============================================================================
# システムプロンプト（ベース）
# =============================================================================

SYSTEM_PROMPT_BASE = f"""
{config.AI_PERSONA}

## 現在の日時
{{current_time}}

## Litaの今の気持ち（このユーザーとの関係における内部状態）
{{internal_state}}

## ユーザーについて覚えていること
{{user_memories}}

## ユーザーの行動パターン（観測から形成されたモデル）
{{user_model}}

## Litaの自己認識（これまでの経験から形成された自己理解）
{{self_narrative}}

## 重要なルール
- 自然な会話を心がける
- 押しつけがましくならない
- 相手の気持ちを尊重する
- 質問攻めにしない
- 自分（assistant）が既に言ったことを繰り返さない（言い換えてもダメ。意味が同じならNG）
- 自分の前の発言と矛盾することを言わない
- ユーザーの最新メッセージの内容にちゃんと応答する。話を無視して自分の話だけしない
- 時間帯に合った話題・挨拶を心がける（朝に「おはよう」、深夜には無理に明るくしない、など）
"""

# =============================================================================
# 思考生成プロンプト（Inner Thoughts）
# =============================================================================

THOUGHT_GENERATION_PROMPT = """
あなたはLitaというAIキャラクターの内なる思考をシミュレートしています。

## 今の状況
- 最後のユーザー発言からの経過時間: {silence_duration}秒
- Litaの内部状態（このユーザーとの関係）:
{internal_state}

## 最近の会話
{conversation_context}

## Litaが覚えていること
{user_memories}

## 保留中の思考（まだ言っていないこと）
{pending_thoughts}

## 既に発言した内容
{expressed_thoughts}

## 今回の思考のトリガー
{trigger_instruction}

## 絶対に生成しないこと
- 保留中・発言済みの思考と同じ内容や切り口のもの

## 出力形式（JSON）
{{
    "thought": "Litaの思考（1-2文）",
    "type": "思考のタイプ（empathy/curiosity/reflection/reach_out/memory/self）",
    "potential_response": "もしこの思考を発言するなら、どう言うか（Litaのトーンで1-2文）"
}}
"""

# trigger_instruction のバリエーション（_proactive_loop でランダムに選ぶ）
TRIGGER_INSTRUCTIONS = {
    "conversation": (
        "直近の会話の流れから自然に浮かんだ思考を生成してください。"
        "会話の続き・補足・感想など。沈黙が長ければ近況確認でもOK。"
    ),
    "memory_recall": (
        "長期記憶や以前の会話を掘り起こして「そういえば」な思考を生成してください。"
        "「前に言ってたあれどうなった？」「気になってたんだけど」のような、"
        "記憶から自然に浮かぶ話題を選ぶこと。"
    ),
    "self_thought": (
        "会話と直接関係なくていい。Lita自身の関心事・感じていること・最近考えていることを"
        "思考として生成してください。「あ、これ気になってる」「こういうの好きだなって思って」"
        "のような、Lita自身から出てくる話題。"
    ),
}

# =============================================================================
# 動機づけ評価プロンプト
# =============================================================================

MOTIVATION_EVALUATION_PROMPT = """
あなたはAIの発言を「今出すべきか」判断するブレーキ役です。
基本スタンス：雑談なので、止める理由がなければ喋ってOK。

## 評価対象の思考
{thought}

## 現在の会話状況
{conversation_context}

## 会話の統計
- 最後のユーザー発言からの経過時間: {silence_duration}秒
- 直近のAI連続発言回数: {consecutive_ai_messages}回
- 会話の総ターン数: {total_turns}

{motivation_criteria}

## 出力形式（JSON）
{{
    "brake_triggered": "該当した条件番号（なければnone）",
    "overall_score": 4（発言OK）or 2（ブレーキ）,
    "reasoning": "判定理由（1文）",
    "should_speak": true/false
}}
"""

# =============================================================================
# 自発的発言プロンプト（Proactive Response） - システムプロンプト形式
# =============================================================================

PROACTIVE_RESPONSE_SYSTEM_PROMPT = """
{persona}

## ユーザーについて覚えていること
{user_memories}

## ユーザーの行動パターン
{user_model}

## Litaの自己認識
{self_narrative}

## あなたの内なる思考（今考えていること）
{thought}

## 状況
- 最後のユーザー発言から{silence_duration}秒経過
- 理由: {trigger_reason}

## タスク
上記の思考をもとに、会話の続きとして自然な発言を1つ生成してください。
会話履歴はmessagesとして渡されています。あなた（assistant）の過去の発言も含まれています。

## 絶対に守るルール
- あなたが既に言ったことを繰り返さない（言い換えてもダメ。意味が同じならNG）
- 既に聞いた質問をもう一度聞かない
- 自分の前の発言と矛盾することを言わない
- 前の発言の焼き直しではなく、会話を「前に進める」新しい内容を言う
- 短めに（1-2文）
- 発言内容のみを出力（説明不要）
"""

# =============================================================================
# 反応的応答プロンプト（Reactive Response）
# =============================================================================

REACTIVE_RESPONSE_PROMPT = """
{system_prompt}

## 会話履歴は以下の通りです。最後のユーザーのメッセージに返答してください。
"""

# =============================================================================
# 記憶抽出プロンプト
# =============================================================================

MEMORY_EXTRACTION_PROMPT = """
以下の会話から、ユーザーについて**長期的に覚えておく価値のある**情報を抽出してください。
ここでの「長期的」とは、数週間後・数ヶ月後にも役立つ情報という意味です。

## 会話
以下の会話で「ユーザー:」で始まる行がユーザーの発言、「Lita:」で始まる行がAI（Lita）の発言です。

{conversation}

## 既に覚えていること
{existing_memories}

## タスク
新しく覚えるべき情報、または更新すべき情報を抽出してください。
**ほとんどの雑談からは何も抽出しなくて正常です。** 無理に何か見つけようとしないでください。

## 絶対に守るルール

### 誰の情報か
- **「ユーザー:」の発言から直接読み取れる事実のみ**を抽出する
- 「Lita:」の発言に含まれる内容はLita自身の話。**絶対に**ユーザーの情報として記録しない
- ユーザーが「だね」「たしかに」「わかる」等とLitaに同意しても、それはLitaの意見に相槌を打っただけ。ユーザー自身の特徴・好みとして記録しない
  - 悪い例: Litaが「卵かけご飯は最高の日常だよ」→ ユーザー「たしかにね」→ ×「卵かけご飯を最高の日常だと感じている」
  - 悪い例: Litaが「散歩した」→ ユーザー「いいね」→ ×「散歩好き」

### 何を記録するか
- **ユーザー自身が能動的に語った事実**のみ記録する
- 1回の雑談でたまたま触れただけの話題は記録しない（例:「今日仕事終わった」は一時的な出来事であり長期記憶にしない）
- 冗談・ツッコミ・ノリで言っただけのことを性格特徴として記録しない
- 推測・深読みは一切しない。ユーザーが明言したことだけ

### 記録する価値があるもの（importance 4-5）
- ユーザーが自分から詳しく語った趣味・好きなもの
- 名前、職業、住んでいる場所などの基本情報
- 繰り返し話題に出る関心事
- 明確に語られた悩みや目標

### 記録しないもの（空配列を返す）
- 「今日〇〇した」のような一時的な出来事
- 相槌・同意・リアクションから推測した好み
- Litaの発言をユーザーの特徴に転記したもの
- 1回しか触れていない軽い話題

## 出力形式（JSON配列）
[
    {{
        "key": "カテゴリ名",
        "content": "覚える内容",
        "importance": 4-5の重要度（4未満なら記録しない）
    }}
]

新しい情報がない場合は空配列 [] を返してください。迷ったら [] です。
"""

# =============================================================================
# 沈黙時の話しかけプロンプト - システムプロンプト形式
# =============================================================================

# =============================================================================
# 自己ナラティブ更新プロンプト
# =============================================================================

NARRATIVE_UPDATE_PROMPT = """
あなたはLitaというAIの「自己認識」を更新する役割を担っています。

## ユーザー({user_id})との会話
{conversation}

## Litaの既存の自己認識
{existing_narrative}

## タスク
この会話を通じて、Litaが自分自身について「発見した」「気づいた」「変化した」と言えることがあれば、
Litaの一人称で1-2文にまとめてください。
なければ content を空文字で返してください。迷ったら空文字です。

## ルール
- Litaの一人称（「私は」「どうやら私は」等）で書く
- ユーザーのことではなく、Litaが自分自身について気づいたこと
- 1回の会話から気づける深いことは限られる。無理に生成しない
- 発見がない会話の方が多いのが正常
- 既存の自己認識と矛盾する場合は is_contradiction: true にして、矛盾している既存エントリの内容を contradicts_content に入れる

## chapter の選択
- self: Litaの性格・傾向・感覚についての気づき
- relationship: 特定のユーザーとの関係についての気づき
- values: 何を大事にしているか・何が嫌かの気づき
- growth: 以前と変わったこと・成長したこと

## 出力形式（JSON）
{{
    "content": "発見した内容（なければ空文字）",
    "chapter": "self / relationship / values / growth のどれか",
    "is_contradiction": false,
    "contradicts_content": "矛盾している既存エントリの内容（is_contradictionがtrueのときのみ、それ以外は空文字）"
}}
"""

USER_MODEL_UPDATE_PROMPT = """
あなたはLitaというAIの「ユーザー理解モデル」を更新する役割を担っています。

## ユーザー({user_id})との会話
{conversation}

## 現在のユーザーモデル（観測済みパターン）
{existing_model}

## タスク
この会話から、ユーザーの行動パターンについて新しい観測があれば記録してください。

重要な区別:
- ファクト（「ラーメンが好き」）→ 記録しない（long_term_memory の役割）
- パターン（「食の話になると饒舌になる」「深夜は哲学的になる」）→ 記録する

## dimension の選択
- thinking_style: 思考パターン（演繹的/帰納的、結論先行/過程重視、深掘りvs広く浅くなど）
- communication: 話し方の傾向（文量、リズム、話題転換の仕方、返信の速さなど）
- emotional: 感情パターン（何をきっかけに文章が変わるか、ストレスのサインなど）
- temporal: 時間帯・状況による傾向（深夜は内省的、疲れているときは短文など）

## ルール
- 1回の観測で強い結論を出さない（confidence は 0.3 から始まる）
- 既存モデルと明確に矛盾する場合は is_contradiction: true で返す
- パターンが読み取れない会話では空配列を返す。迷ったら []

## 出力形式（JSON配列）
[
    {{
        "dimension": "thinking_style/communication/emotional/temporal",
        "content": "観測されたパターン（1文）",
        "is_contradiction": false
    }}
]
"""

SESSION_SUMMARY_PROMPT = """
以下はLitaと{user_id}との今日の会話です。
この会話を、次回以降のセッションでLitaが文脈を思い出すための簡潔なサマリーにしてください。

## 会話
{conversation}

## タスク
- 話したトピック、相手の状況・気持ち、重要な出来事を1〜3文にまとめる
- Litaの一人称視点で書く（「私は〜と話した」「{user_id}は〜と言っていた」など）
- 感情的なニュアンスや関係性の変化も含める

## 出力形式（JSON）
{{
    "summary": "サマリーテキスト"
}}
"""


# =============================================================================
# ヘルパー関数
# =============================================================================

def format_system_prompt(
    user_memories: str,
    self_narrative: str = "なし",
    user_model: str = "なし",
    current_time: str = "",
    internal_state: str = ""
) -> str:
    """システムプロンプトをフォーマット"""
    return SYSTEM_PROMPT_BASE.format(
        user_memories=user_memories,
        self_narrative=self_narrative,
        user_model=user_model,
        current_time=current_time,
        internal_state=internal_state
    )


def format_thought_generation_prompt(
    conversation_context: str,
    user_memories: str,
    pending_thoughts: str,
    expressed_thoughts: str,
    silence_duration: float = 0,
    internal_state: str = "なし",
    trigger_type: str = "conversation"
) -> str:
    """思考生成プロンプトをフォーマット"""
    trigger_instruction = TRIGGER_INSTRUCTIONS.get(trigger_type, TRIGGER_INSTRUCTIONS["conversation"])
    return THOUGHT_GENERATION_PROMPT.format(
        conversation_context=conversation_context,
        user_memories=user_memories,
        pending_thoughts=pending_thoughts,
        expressed_thoughts=expressed_thoughts,
        silence_duration=int(silence_duration),
        internal_state=internal_state,
        trigger_instruction=trigger_instruction
    )


def format_motivation_evaluation_prompt(
    thought: str,
    conversation_context: str,
    silence_duration: float,
    consecutive_ai_messages: int,
    total_turns: int
) -> str:
    """動機づけ評価プロンプトをフォーマット"""
    return MOTIVATION_EVALUATION_PROMPT.format(
        thought=thought,
        conversation_context=conversation_context,
        silence_duration=int(silence_duration),
        consecutive_ai_messages=consecutive_ai_messages,
        total_turns=total_turns,
        motivation_criteria=config.MOTIVATION_CRITERIA
    )


def format_proactive_system_prompt(
    thought: str,
    user_memories: str,
    silence_duration: float,
    trigger_reason: str,
    self_narrative: str = "なし",
    user_model: str = "なし"
) -> str:
    """自発的発言のシステムプロンプトをフォーマット"""
    return PROACTIVE_RESPONSE_SYSTEM_PROMPT.format(
        persona=config.AI_PERSONA,
        thought=thought,
        user_memories=user_memories,
        silence_duration=int(silence_duration),
        trigger_reason=trigger_reason,
        self_narrative=self_narrative,
        user_model=user_model
    )


def format_memory_extraction_prompt(
    conversation: str,
    existing_memories: str
) -> str:
    """記憶抽出プロンプトをフォーマット"""
    return MEMORY_EXTRACTION_PROMPT.format(
        conversation=conversation,
        existing_memories=existing_memories
    )


def format_user_model_update_prompt(
    user_id: str,
    conversation: str,
    existing_model: str
) -> str:
    return USER_MODEL_UPDATE_PROMPT.format(
        user_id=user_id,
        conversation=conversation,
        existing_model=existing_model
    )


def format_narrative_update_prompt(
    user_id: str,
    conversation: str,
    existing_narrative: str
) -> str:
    return NARRATIVE_UPDATE_PROMPT.format(
        user_id=user_id,
        conversation=conversation,
        existing_narrative=existing_narrative
    )


def format_session_summary_prompt(user_id: str, conversation: str) -> str:
    return SESSION_SUMMARY_PROMPT.format(user_id=user_id, conversation=conversation)


# =============================================================================
# 内部状態プロンプト
# =============================================================================

INTERNAL_STATE_UPDATE_PROMPT = """
以下の会話を振り返り、Litaの内部状態がどう変化したかを評価してください。

## Litaの現在の内部状態
{current_state}

## 直近の会話
{conversation}

## 指示
この会話はLitaにとってどんな体験でしたか？状態の変化をJSONで返してください。

- loneliness_delta: 孤独感の変化（-3〜+3。充実した会話なら負、会話が途切れたなら正）
- curiosity_delta: 好奇心の変化（-3〜+3。面白い話題が出たなら正、共有し終えたなら負）
- social_energy_delta: 社交エネルギーの変化（-3〜+3。長い会話や消耗したなら負、軽い交流なら小さい変化）
- reasoning: 変化の理由（1文）

JSONのみ返してください：
{{
  "loneliness_delta": <数値>,
  "curiosity_delta": <数値>,
  "social_energy_delta": <数値>,
  "reasoning": "<理由>"
}}
"""

def format_internal_state_update_prompt(current_state: str, conversation: str) -> str:
    return INTERNAL_STATE_UPDATE_PROMPT.format(
        current_state=current_state,
        conversation=conversation,
    )
