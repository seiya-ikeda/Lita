"""
Proactive AI Friend - Slack Bot
Inner ThoughtsフレームワークをSlackに移植
"""

import asyncio
import re
import uuid
from datetime import datetime

from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

import config
from memory import MemoryManager
from inner_thoughts import InnerThoughtsEngine
from research_logger import ResearchLogger
from information_gatherer import InformationGatherer
from response_classifier import ResponseClassifier


class ProactiveAISlackBot:
    """
    Proactive AI Friend Slack Bot

    機能:
    - ユーザーメッセージへの反応的応答
    - Inner Thoughtsによる自発的発言
    - 長期記憶によるユーザー理解
    - 研究用ログ収集
    """

    def __init__(self):
        self.app = AsyncApp(token=config.SLACK_BOT_TOKEN)

        # エンジン
        self.engine = InnerThoughtsEngine()
        self.info_gatherer = InformationGatherer()
        self.classifier = ResponseClassifier()

        # ユーザーごとの記憶管理
        self.memories: dict[str, MemoryManager] = {}

        # セッション管理
        self.session_id = str(uuid.uuid4())[:8]
        self.logger = ResearchLogger(self.session_id)

        # ユーザーごとの会話チャンネル（DM channel ID）
        self.user_channels: dict[str, str] = {}

        # 処理中フラグ（二重応答防止）
        self.processing: set[str] = set()

        # 情報収集の最終検索時刻
        self.last_info_search: dict[str, datetime] = {}

        # Bot自身のユーザーID（起動時に取得）
        self.bot_user_id: str = ""

        # イベントハンドラとコマンドを登録
        self._register_handlers()

    # =========================================================================
    # ハンドラ登録
    # =========================================================================

    def _register_handlers(self):
        """イベントハンドラとスラッシュコマンドを登録"""

        # メッセージイベント
        @self.app.event("message")
        async def handle_message(event, say, client):
            await self._on_message(event, say, client)

        # スラッシュコマンド
        @self.app.command("/lita-status")
        async def handle_status(ack, command, client):
            await ack()
            await self._cmd_status(command, client)

        @self.app.command("/lita-memories")
        async def handle_memories(ack, command, client):
            await ack()
            await self._cmd_memories(command, client)

        @self.app.command("/lita-forget")
        async def handle_forget(ack, command, client):
            await ack()
            await self._cmd_forget(command, client)

        @self.app.command("/lita-thoughts")
        async def handle_thoughts(ack, command, client):
            await ack()
            await self._cmd_thoughts(command, client)

        @self.app.command("/lita-config")
        async def handle_config(ack, command, client):
            await ack()
            await self._cmd_config(command, client)

        @self.app.command("/lita-interests")
        async def handle_interests(ack, command, client):
            await ack()
            await self._cmd_interests(command, client)

        @self.app.command("/lita-search")
        async def handle_search(ack, command, client):
            await ack()
            await self._cmd_search(command, client)

        @self.app.command("/lita-export")
        async def handle_export(ack, command, client):
            await ack()
            await self._cmd_export(command, client)

    # =========================================================================
    # メッセージ処理
    # =========================================================================

    async def _on_message(self, event, say, client):
        """メッセージ受信時の処理"""
        # Bot自身のメッセージ、サブタイプ付き（編集・削除等）は無視
        if event.get("bot_id") or event.get("subtype"):
            return

        user_id = event.get("user", "")
        text = event.get("text", "")
        channel = event.get("channel", "")
        channel_type = event.get("channel_type", "")
        ts = event.get("ts", "")

        # DMまたはメンションされた場合のみ反応
        is_dm = channel_type == "im"
        is_mentioned = f"<@{self.bot_user_id}>" in text

        if not (is_dm or is_mentioned):
            return

        # メンションを除去
        content = re.sub(r"<@[A-Z0-9]+>", "", text).strip()
        if not content:
            return

        # 二重処理防止
        process_key = f"{user_id}:{ts}"
        if process_key in self.processing:
            return
        self.processing.add(process_key)

        try:
            # 記憶マネージャー取得または作成
            memory = self._get_memory(user_id)

            # ユーザーとチャンネルの対応を保存
            self.user_channels[user_id] = channel

            # ユーザーメッセージを記録
            memory.add_message("user", content)
            self.logger.log_user_message(user_id, content)

            # 応答タイプを判定
            decision = await self.classifier.classify(
                content,
                memory.get_context_summary()
            )

            if decision.action == "react" and decision.reaction:
                # リアクションで十分な場合
                await client.reactions_add(
                    channel=channel,
                    timestamp=ts,
                    name=self._sanitize_reaction_name(decision.reaction)
                )
                self.logger.log_ai_response(
                    user_id,
                    f"[reaction: {decision.reaction}]",
                    is_proactive=False,
                    metadata={"type": "reaction", "reason": decision.reason}
                )

            elif decision.action == "reply":
                # 返信を生成
                response = await self.engine.generate_reactive_response(memory)

                is_fallback = "調子悪いみたい..." in response
                if not is_fallback:
                    memory.add_message("assistant", response)
                self.logger.log_ai_response(user_id, response, is_proactive=False)

                await client.chat_postMessage(
                    channel=channel,
                    text=response
                )

            # 定期的に記憶を抽出（5ターンごと）
            if len(memory.short_term) >= 5 and len(memory.short_term) % 5 <= 1:
                await self._extract_and_save_memories(memory)

        finally:
            self.processing.discard(process_key)

    # =========================================================================
    # Proactiveサイクル
    # =========================================================================

    async def _proactive_loop(self):
        """定期的に実行されるProactiveサイクル"""
        while True:
            await asyncio.sleep(30)

            for user_id, memory in list(self.memories.items()):
                try:
                    result = await self.engine.process_proactive_cycle(memory)

                    if result is None:
                        continue

                    # 思考ログを記録（発言の有無に関わらず）
                    if result.get("thought_content"):
                        self.logger.log_thought(
                            user_id=user_id,
                            thought_content=result["thought_content"],
                            trigger_reason=result["trigger_reason"],
                            motivation_score=result["motivation_score"],
                            evaluation_details=result["evaluation"],
                            was_expressed=result["was_expressed"],
                            response_if_expressed=result.get("response"),
                        )

                    # 発言がある場合のみ送信
                    response = result.get("response")
                    if response:
                        channel = self.user_channels.get(user_id)
                        if channel:
                            memory.add_message("assistant", response)
                            self.logger.log_ai_response(
                                user_id, response, is_proactive=True,
                                metadata={
                                    "trigger": result["trigger_reason"],
                                    "motivation_score": result["motivation_score"],
                                }
                            )

                            await self.app.client.chat_postMessage(
                                channel=channel,
                                text=response
                            )

                except Exception as e:
                    print(f"Proactive cycle error for {user_id}: {e}")

    # =========================================================================
    # 情報収集サイクル
    # =========================================================================

    async def _information_gathering_loop(self):
        """定期的に実行される情報収集サイクル"""
        while True:
            await asyncio.sleep(30 * 60)  # 30分間隔

            if not config.BRAVE_SEARCH_API_KEY:
                continue

            now = datetime.now()

            for user_id, memory in list(self.memories.items()):
                try:
                    last_search = self.last_info_search.get(user_id)
                    if last_search:
                        elapsed = (now - last_search).total_seconds()
                        if elapsed < config.SEARCH_INTERVAL:
                            continue

                    if not memory.long_term:
                        continue

                    if not memory.can_intervene():
                        continue

                    result = await self.info_gatherer.find_shareable_article(memory)

                    if result:
                        article, message = result
                        channel = self.user_channels.get(user_id)
                        if channel:
                            memory.add_message("assistant", message)
                            self.logger.log_ai_response(
                                user_id, message, is_proactive=True,
                                metadata={
                                    "trigger": "information_share",
                                    "article_title": article.title,
                                    "article_url": article.url,
                                    "relevance_score": article.relevance_score
                                }
                            )

                            await self.app.client.chat_postMessage(
                                channel=channel,
                                text=message
                            )

                    self.last_info_search[user_id] = now

                except Exception as e:
                    print(f"Information gathering error for {user_id}: {e}")

    # =========================================================================
    # スラッシュコマンド
    # =========================================================================

    async def _cmd_status(self, command, client):
        """現在のステータスを表示"""
        user_id = command["user_id"]
        channel = command["channel_id"]
        memory = self._get_memory(user_id)

        stats = self.logger.get_thought_statistics()
        metrics = self.logger.calculate_metrics(user_id)

        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"{config.AI_NAME} Status"}
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Session ID:* `{self.session_id}`"},
                    {"type": "mrkdwn", "text": f"*Condition:* `{config.EXPERIMENT_CONDITION}`"},
                ]
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Total turns:* {metrics.total_turns}"},
                    {"type": "mrkdwn", "text": f"*Your messages:* {metrics.user_messages}"},
                    {"type": "mrkdwn", "text": f"*My replies:* {metrics.ai_reactive_responses}"},
                    {"type": "mrkdwn", "text": f"*My initiatives:* {metrics.ai_proactive_interventions}"},
                ]
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Short-term memory:* {len(memory.short_term)} turns"},
                    {"type": "mrkdwn", "text": f"*Long-term memory:* {len(memory.long_term)} items"},
                    {"type": "mrkdwn", "text": f"*Pending thoughts:* {len(memory.thought_reservoir)}"},
                ]
            },
        ]

        if stats:
            blocks.append({
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Generated thoughts:* {stats.get('total_thoughts', 0)}"},
                    {"type": "mrkdwn", "text": f"*Expressed thoughts:* {stats.get('expressed_thoughts', 0)}"},
                    {"type": "mrkdwn", "text": f"*Avg motivation:* {stats.get('avg_motivation_score', 0):.2f}"},
                ]
            })

        await client.chat_postEphemeral(
            channel=channel, user=user_id, blocks=blocks,
            text=f"{config.AI_NAME} Status"
        )

    async def _cmd_memories(self, command, client):
        """覚えていることを表示"""
        user_id = command["user_id"]
        channel = command["channel_id"]
        memory = self._get_memory(user_id)

        memories_text = memory.get_all_memories_summary()

        await client.chat_postEphemeral(
            channel=channel, user=user_id,
            text=f"*{config.AI_NAME} が覚えていること:*\n{memories_text if memories_text else 'まだ何も覚えていません'}"
        )

    async def _cmd_forget(self, command, client):
        """記憶をリセット"""
        user_id = command["user_id"]
        channel = command["channel_id"]

        if user_id in self.memories:
            del self.memories[user_id]

        await client.chat_postEphemeral(
            channel=channel, user=user_id,
            text="記憶をリセットしました。また一から仲良くなりましょう！"
        )

    async def _cmd_thoughts(self, command, client):
        """保留中の思考を表示"""
        user_id = command["user_id"]
        channel = command["channel_id"]
        memory = self._get_memory(user_id)

        pending = memory.get_pending_thoughts()

        if not pending:
            await client.chat_postEphemeral(
                channel=channel, user=user_id,
                text="今のところ、特に言いたいことはないかな"
            )
            return

        threshold = config.MOTIVATION_THRESHOLD
        lines = []
        for i, thought in enumerate(pending[:5], 1):
            passed = "passed" if thought.motivation_score >= threshold else "below"
            line = f"*Thought {i}* (score: {thought.motivation_score:.1f})\n"
            line += f"> {thought.content[:100]}\n"
            if thought.reasoning:
                line += f"> Reason: {thought.reasoning[:100]}\n"
            line += f"> Threshold({threshold}): {passed}\n"
            lines.append(line)

        await client.chat_postEphemeral(
            channel=channel, user=user_id,
            text="\n".join(lines)
        )

    async def _cmd_config(self, command, client):
        """設定を表示"""
        user_id = command["user_id"]
        channel = command["channel_id"]

        text = (
            f"*Proactive Settings*\n"
            f"- Motivation threshold: {config.MOTIVATION_THRESHOLD}\n"
            f"- Thought interval: {config.THOUGHT_GENERATION_INTERVAL}s\n"
            f"- Silence timeout: {config.SILENCE_TIMEOUT}s\n"
            f"- Max consecutive: {config.MAX_CONSECUTIVE_INTERVENTIONS}\n\n"
            f"*Information Gathering*\n"
            f"- Enabled: {config.ENABLE_INFORMATION_GATHERING}\n"
            f"- Search interval: {config.SEARCH_INTERVAL}s\n"
            f"- Share threshold: {config.INFO_SHARE_MOTIVATION_THRESHOLD}\n"
            f"- Max daily shares: {config.MAX_DAILY_SHARES}"
        )

        await client.chat_postEphemeral(
            channel=channel, user=user_id, text=text
        )

    async def _cmd_interests(self, command, client):
        """推測された興味を表示"""
        user_id = command["user_id"]
        channel = command["channel_id"]
        memory = self._get_memory(user_id)

        interests = await self.info_gatherer.extract_interests(memory)

        if interests:
            text = "*あなたの興味（推測）:*\n" + "\n".join([f"- {i}" for i in interests])
        else:
            text = "まだ興味を把握できていません。もっとお話ししましょう！"

        await client.chat_postEphemeral(
            channel=channel, user=user_id, text=text
        )

    async def _cmd_search(self, command, client):
        """今すぐ情報を探して共有"""
        user_id = command["user_id"]
        channel = command["channel_id"]

        if not config.BRAVE_SEARCH_API_KEY:
            await client.chat_postEphemeral(
                channel=channel, user=user_id,
                text="検索機能が設定されていません（BRAVE_SEARCH_API_KEY が必要です）"
            )
            return

        memory = self._get_memory(user_id)

        await client.chat_postEphemeral(
            channel=channel, user=user_id,
            text="あなたが興味ありそうな情報を探しています..."
        )

        result = await self.info_gatherer.find_shareable_article(memory)

        if result:
            article, message = result

            memory.add_message("assistant", message)
            self.logger.log_ai_response(
                user_id, message, is_proactive=True,
                metadata={
                    "trigger": "manual_search",
                    "article_title": article.title,
                    "relevance_score": article.relevance_score
                }
            )

            await client.chat_postMessage(channel=channel, text=message)
        else:
            await client.chat_postEphemeral(
                channel=channel, user=user_id,
                text="今回は特に良さそうな情報が見つからなかったかも...また後で探してみるね！"
            )

    async def _cmd_export(self, command, client):
        """ログをエクスポート"""
        user_id = command["user_id"]
        channel = command["channel_id"]

        self.logger.save_session_summary()
        summary = self.logger.export_session_summary()

        await client.chat_postEphemeral(
            channel=channel, user=user_id,
            text=f"*Log exported*\nSaved to: `{config.LOG_DIRECTORY}/`\n```{str(summary)[:500]}...```"
        )

    # =========================================================================
    # ヘルパー関数
    # =========================================================================

    def _get_memory(self, user_id: str) -> MemoryManager:
        """ユーザーの記憶マネージャーを取得または作成"""
        if user_id not in self.memories:
            self.memories[user_id] = MemoryManager(user_id)
        return self.memories[user_id]

    @staticmethod
    def _sanitize_reaction_name(name: str) -> str:
        """LLMが返したリアクション名をサニタイズ"""
        # コロンや余分な空白を除去（:thumbsup: → thumbsup）
        return name.strip().strip(":")

    async def _extract_and_save_memories(self, memory: MemoryManager):
        """記憶を抽出して保存（importance 4以上のみ）"""
        try:
            new_memories = await self.engine.extract_memories(memory)
            for mem in new_memories:
                importance = mem.get("importance", 0)
                if importance >= 4:
                    memory.add_long_term_memory(
                        key=mem.get("key", "その他"),
                        content=mem.get("content", ""),
                        importance=importance
                    )
                else:
                    print(f"[DEBUG] Memory skipped (importance {importance} < 4): {mem.get('content', '')[:50]}")
        except Exception as e:
            print(f"Memory extraction error: {e}")

    # =========================================================================
    # 起動
    # =========================================================================

    async def start(self):
        """Botを起動"""
        # Bot自身のユーザーIDを取得
        auth = await self.app.client.auth_test()
        self.bot_user_id = auth["user_id"]

        print(f"{config.AI_NAME} がオンラインになりました！ (Slack)")
        print(f"Session ID: {self.session_id}")
        print(f"Condition: {config.EXPERIMENT_CONDITION}")
        print(f"Bot User ID: {self.bot_user_id}")
        print("-" * 50)

        # バックグラウンドタスクを起動
        if config.EXPERIMENT_CONDITION == "proactive":
            asyncio.create_task(self._proactive_loop())

        if config.ENABLE_INFORMATION_GATHERING:
            asyncio.create_task(self._information_gathering_loop())

        # Socket Modeで接続
        handler = AsyncSocketModeHandler(self.app, config.SLACK_APP_TOKEN)
        await handler.start_async()


# =============================================================================
# メイン
# =============================================================================

def main():
    """Botを起動"""
    if not config.SLACK_BOT_TOKEN:
        print("SLACK_BOT_TOKEN が設定されていません")
        print("  .env ファイルに SLACK_BOT_TOKEN=xoxb-... を追加してください")
        return

    if not config.SLACK_APP_TOKEN:
        print("SLACK_APP_TOKEN が設定されていません")
        print("  .env ファイルに SLACK_APP_TOKEN=xapp-... を追加してください")
        return

    if not config.OPENAI_API_KEY:
        print("OPENAI_API_KEY が設定されていません")
        return

    bot = ProactiveAISlackBot()
    asyncio.run(bot.start())


if __name__ == "__main__":
    main()
