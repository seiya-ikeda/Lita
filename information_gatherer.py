"""
Proactive AI Friend - Information Gatherer
ユーザーの興味に基づいて外部情報を収集し、共有する
"""

import aiohttp
import json
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, asdict
from openai import AsyncOpenAI

import config
from memory import MemoryManager


@dataclass
class Article:
    """収集した記事/情報"""
    title: str
    url: str
    description: str
    source: str
    published: Optional[str]
    search_query: str  # どの興味から見つかったか
    found_at: str
    relevance_score: float = 0.0
    shared: bool = False
    
    def to_dict(self):
        return asdict(self)


class InformationGatherer:
    """
    ユーザーの興味に基づいて情報を収集するクラス
    
    機能:
    - 長期記憶から興味を抽出
    - Brave Searchで検索
    - 関連性をスコアリング
    - 自然な共有メッセージを生成
    """
    
    def __init__(self):
        self.client = AsyncOpenAI(
            base_url=config.LLM_BASE_URL,
            api_key=config.OPENAI_API_KEY or "no-key"
        )
        self.brave_api_key = config.BRAVE_SEARCH_API_KEY
        
        # ユーザーごとの記事キャッシュ（既に見つけた記事のURL）
        self.seen_urls: dict[str, set[str]] = {}
        
        # ユーザーごとの本日の共有数
        self.daily_shares: dict[str, int] = {}
        self.last_share_reset: dict[str, datetime] = {}
    
    # =========================================================================
    # 興味の抽出
    # =========================================================================
    
    async def extract_interests(self, memory: MemoryManager) -> list[str]:
        """
        長期記憶からユーザーの興味を抽出
        """
        memories_summary = memory.get_all_memories_summary()
        
        if not memories_summary or memories_summary == "まだユーザーについての情報がありません。":
            return []
        
        prompt = f"""
以下のユーザー情報から、検索に使えそうな「興味・関心」を抽出してください。

## ユーザー情報
{memories_summary}

## タスク
- ユーザーが興味を持っていそうなトピックを3-5個抽出
- 検索クエリとして使えるように、具体的なキーワードで
- 一般的すぎるもの（音楽、映画など）は避け、具体的に

## 出力形式（JSON配列）
["キーワード1", "キーワード2", "キーワード3"]

例: ["Apple 新製品", "機械学習 最新研究", "京都 カフェ"]
"""
        
        try:
            response = await self.client.chat.completions.create(
                model=config.LLM_MODEL,
                max_completion_tokens=config.MAX_COMPLETION_TOKENS,
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = response.choices[0].message.content
            # JSON抽出
            import re
            match = re.search(r'\[.*?\]', text, re.DOTALL)
            if match:
                return json.loads(match.group())
            return []
            
        except Exception as e:
            print(f"Interest extraction error: {e}")
            return []
    
    # =========================================================================
    # 検索
    # =========================================================================
    
    async def search_brave(self, query: str) -> list[dict]:
        """
        Brave Search APIで検索
        """
        if not self.brave_api_key:
            print("Brave Search API key not configured")
            return []
        
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "X-Subscription-Token": self.brave_api_key,
            "Accept": "application/json"
        }
        params = {
            "q": query,
            "count": config.SEARCH_RESULTS_COUNT,
            "search_lang": config.SEARCH_LANGUAGE,
            "freshness": "pw"  # 過去1週間
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("web", {}).get("results", [])
                    else:
                        print(f"Brave Search error: {resp.status}")
                        return []
        except Exception as e:
            print(f"Brave Search request error: {e}")
            return []
    
    async def search_for_user(
        self, 
        memory: MemoryManager, 
        interests: Optional[list[str]] = None
    ) -> list[Article]:
        """
        ユーザーの興味に基づいて検索し、新しい記事を返す
        """
        user_id = memory.user_id
        
        # 興味がなければ抽出
        if not interests:
            interests = await self.extract_interests(memory)
        
        if not interests:
            return []
        
        # 既知URLのセット
        if user_id not in self.seen_urls:
            self.seen_urls[user_id] = set()
        
        articles = []
        
        import asyncio
        for i, interest in enumerate(interests[:3]):  # 最大3つの興味で検索
            if i > 0:
                # 無料プランのレート制限対策: 2秒待機
                await asyncio.sleep(2)
            
            results = await self.search_brave(interest)
            
            for result in results:
                url = result.get("url", "")
                
                # 既に見つけたURLはスキップ
                if url in self.seen_urls[user_id]:
                    continue
                
                self.seen_urls[user_id].add(url)
                
                article = Article(
                    title=result.get("title", ""),
                    url=url,
                    description=result.get("description", ""),
                    source=result.get("meta_url", {}).get("hostname", ""),
                    published=result.get("age", None),
                    search_query=interest,
                    found_at=datetime.now().isoformat()
                )
                articles.append(article)
        
        return articles
    
    # =========================================================================
    # 関連性評価
    # =========================================================================
    
    async def evaluate_article_relevance(
        self, 
        article: Article, 
        memory: MemoryManager
    ) -> float:
        """
        記事がユーザーにとってどれくらい関連性があるかスコアリング
        """
        memories_summary = memory.get_all_memories_summary()
        conversation_context = memory.get_context_summary()
        
        prompt = f"""
以下の記事を、このユーザーに共有すべきかどうか評価してください。

## 記事
タイトル: {article.title}
説明: {article.description}
ソース: {article.source}
検索クエリ: {article.search_query}

## ユーザー情報
{memories_summary}

## 最近の会話
{conversation_context}

## 評価基準（各1-5点）
1. 関連性: ユーザーの興味とどれくらいマッチするか
2. 新鮮さ: 新しい情報・発見がありそうか
3. 会話価値: これを共有したら会話が盛り上がりそうか
4. 信頼性: ソースは信頼できそうか
5. タイミング: 今共有するのは適切か

## 出力形式（JSON）
{{
    "relevance": 1-5,
    "freshness": 1-5,
    "conversation_value": 1-5,
    "reliability": 1-5,
    "timing": 1-5,
    "overall_score": 1-5の総合評価,
    "reasoning": "評価理由（1文）"
}}
"""
        
        try:
            response = await self.client.chat.completions.create(
                model=config.LLM_MODEL,
                max_completion_tokens=config.MAX_COMPLETION_TOKENS,
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = response.choices[0].message.content
            # JSON抽出
            import re
            match = re.search(r'\{.*?\}', text, re.DOTALL)
            if match:
                result = json.loads(match.group())
                return result.get("overall_score", 0)
            return 0
            
        except Exception as e:
            print(f"Article evaluation error: {e}")
            return 0
    
    # =========================================================================
    # 共有メッセージ生成
    # =========================================================================
    
    async def generate_share_message(
        self, 
        article: Article, 
        memory: MemoryManager
    ) -> str:
        """
        記事を自然に共有するメッセージを生成
        """
        memories_summary = memory.get_all_memories_summary()
        
        prompt = f"""
友達に面白い記事を教えるような感じで、以下の情報を共有するメッセージを作ってください。

## 共有する記事
タイトル: {article.title}
説明: {article.description}
URL: {article.url}

## ユーザーについて
{memories_summary}

## 検索のきっかけ
「{article.search_query}」に興味があると思って探してみた

## 重要なルール
- 押し付けがましくなく、自然に
- 「ねえねえ」「そういえば」「見つけたんだけど」など自然な導入
- URLは必ず含める
- 2-3文程度で短く
- 相手の興味に関連付けて紹介
- 質問で終わらなくてもOK

## 出力
メッセージ本文のみ（説明不要）
"""
        
        try:
            response = await self.client.chat.completions.create(
                model=config.LLM_MODEL,
                max_completion_tokens=config.MAX_COMPLETION_TOKENS,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.choices[0].message.content
            return content.strip() if content else ""
            
        except Exception as e:
            print(f"Share message generation error: {e}")
            return ""
    
    # =========================================================================
    # メインフロー
    # =========================================================================
    
    async def find_shareable_article(
        self, 
        memory: MemoryManager
    ) -> Optional[tuple[Article, str]]:
        """
        共有すべき記事を探し、メッセージを生成
        
        Returns:
            (Article, share_message) or None
        """
        user_id = memory.user_id
        
        # 1日の共有制限チェック
        if not self._can_share_today(user_id):
            return None
        
        # 記事を検索
        articles = await self.search_for_user(memory)
        
        if not articles:
            return None
        
        # 各記事を評価
        best_article = None
        best_score = 0
        
        for article in articles:
            score = await self.evaluate_article_relevance(article, memory)
            article.relevance_score = score
            
            if score > best_score and score >= config.INFO_SHARE_MOTIVATION_THRESHOLD:
                best_score = score
                best_article = article
        
        if not best_article:
            return None
        
        # メッセージ生成
        message = await self.generate_share_message(best_article, memory)
        
        if message:
            best_article.shared = True
            self._increment_daily_shares(user_id)
            return (best_article, message)
        
        return None
    
    # =========================================================================
    # ヘルパー
    # =========================================================================
    
    def _can_share_today(self, user_id: str) -> bool:
        """今日まだ共有できるかチェック"""
        now = datetime.now()
        
        # 日付が変わっていたらリセット
        if user_id in self.last_share_reset:
            if now.date() > self.last_share_reset[user_id].date():
                self.daily_shares[user_id] = 0
                self.last_share_reset[user_id] = now
        else:
            self.last_share_reset[user_id] = now
            self.daily_shares[user_id] = 0
        
        return self.daily_shares.get(user_id, 0) < config.MAX_DAILY_SHARES
    
    def _increment_daily_shares(self, user_id: str):
        """共有カウントを増やす"""
        self.daily_shares[user_id] = self.daily_shares.get(user_id, 0) + 1
    
    def get_share_stats(self, user_id: str) -> dict:
        """共有統計を取得"""
        return {
            "today_shares": self.daily_shares.get(user_id, 0),
            "max_daily": config.MAX_DAILY_SHARES,
            "seen_articles": len(self.seen_urls.get(user_id, set()))
        }
