"""Play the actual competition policy through a local browser table.

The event rules and bot runtime are imported unchanged. Each browser has an
isolated private deal stream and opponent-learning memory. This exhibition
has no thinking bank and does not participate in the overnight search.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import logging
from pathlib import Path
import secrets
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn

from evaluator import card_to_str, category_name
from equity_engine import score_hand
from game import Action, AgentConfig, BIG_BLIND, Game, STARTING_STACK, button_for_deal, make_deal
import strategy

LOGGER = logging.getLogger(__name__)
COOKIE = "jr007_poker_session"
SESSION_TTL_SECONDS = 6 * 60 * 60
MAX_SESSIONS = 128
FRONTEND = Path(__file__).with_suffix("") / "index.html"


def card_text(cards):
    return [card_to_str(card) for card in cards]


def category(hole, board):
    return category_name(score_hand(hole, board)) if len(board) >= 3 else None


class ActionBody(BaseModel):
    request_id: str = Field(min_length=1, max_length=100)
    action: str = Field(min_length=1, max_length=30)


class NextBody(BaseModel):
    request_id: str = Field(min_length=1, max_length=100)


class HumanSession:
    def __init__(self, hands=1000, bot=strategy):
        self.lock = threading.RLock()
        self.touched = time.monotonic()
        self._seed = secrets.token_hex(32)
        self.bot = bot
        self.hands = hands
        self.memory = {}
        self.bot.new_half(AgentConfig(num_hands=hands), self.memory)
        self.hand_index = 0
        self.human_seat = 0
        self.score = {"human_chips": 0, "bot_chips": 0, "hands_played": 0,
                      "wins": 0, "losses": 0, "ties": 0}
        self.request_id = ""
        self.error = None
        self._settled = False
        self._start_hand()

    def _rotate_request(self):
        self.request_id = secrets.token_urlsafe(12)

    def _start_hand(self):
        self.game = Game(make_deal(self._seed, self.hand_index),
                         button_for_deal(self.hand_index), self.hand_index)
        self._settled = False
        self._rotate_request()
        self._advance_bot()

    def _advance_bot(self):
        try:
            while not self.game.is_terminal and self.game.to_act != self.human_seat:
                observation = self.game.observation(1 - self.human_seat)
                choice = self.bot.choose_action(observation, self.memory)
                if not isinstance(choice, Action) or choice not in observation.legal_actions:
                    raise RuntimeError("Bot returned an illegal action")
                self.game.apply(choice)
            if self.game.is_terminal and not self._settled:
                result = self.game.hand_result(self.human_seat)
                self.score["human_chips"] += result.my_net
                self.score["bot_chips"] -= result.my_net
                self.score["hands_played"] += 1
                self.score[{"me": "wins", "opp": "losses", "split": "ties"}[result.winner]] += 1
                self._settled = True
                self.bot.on_hand_end(self.game.hand_result(1 - self.human_seat), self.memory)
        except Exception:
            LOGGER.exception("Human-play bot callback failed")
            self.error = "The bot could not finish this hand. Start a new session to try again."

    def _check_request(self, request_id):
        if not request_id.isascii() or not secrets.compare_digest(request_id, self.request_id):
            raise HTTPException(409, "The table changed. Refresh the hand before acting again.")
        if self.error:
            raise HTTPException(503, self.error)

    def act(self, request_id, action):
        self._check_request(request_id)
        if self.game.is_terminal:
            raise HTTPException(409, "This hand is over. Deal the next hand.")
        if self.game.to_act != self.human_seat:
            raise HTTPException(409, "It is not your turn.")
        try:
            choice = Action[action]
        except KeyError:
            raise HTTPException(400, "Unknown poker action.") from None
        if choice not in self.game.legal_actions():
            raise HTTPException(400, "That action is not legal for this hand.")
        self.game.apply(choice)
        self._rotate_request()
        self._advance_bot()
        return self.state()

    def next_hand(self, request_id):
        self._check_request(request_id)
        if not self.game.is_terminal:
            raise HTTPException(409, "Finish this hand before dealing another.")
        if self.score["hands_played"] >= self.hands:
            raise HTTPException(409, "This session is complete. Start a new session to keep playing.")
        self.hand_index += 1
        self._start_hand()
        return self.state()

    def state(self):
        # Serialize a legal player observation rather than any Game/Deal data.
        observation = self.game.observation(self.human_seat)
        public = observation.to_dict()
        public.update(hole_text=card_text(observation.hole_cards),
                      board_text=card_text(observation.board),
                      current_made_hand_category=category(observation.hole_cards, observation.board))
        result = None
        bot_cards = None
        if self.game.is_terminal:
            own = self.game.hand_result(self.human_seat)
            result = own.to_dict()
            result.update(human_net=own.my_net, bot_net=-own.my_net,
                          human_category=category(observation.hole_cards, own.board),
                          bot_category=category(own.opp_hole, own.board) if own.opp_hole is not None else None)
            bot_cards = card_text(own.opp_hole) if own.opp_hole is not None else None
        history = [{"street": record.street.name,
                    "actor": "you" if record.seat == self.human_seat else "bot",
                    "action": record.action.name, "amount": record.amount,
                    "street_total": record.street_total} for record in self.game.history]
        return {"request_id": self.request_id, "hand_number": self.hand_index + 1,
                "hand_index": self.hand_index, "hand_over": self.game.is_terminal,
                "session_complete": self.score["hands_played"] >= self.hands,
                "hands_per_session": self.hands, "bot_name": "jr007",
                "observation": public, "bot_cards": bot_cards, "result": result,
                "score": dict(self.score), "history": history, "error": self.error,
                "rules": {"deck_ranks": "6 through ace", "hole_cards": 5,
                          "use_hole_cards": 2, "use_board_cards": 3,
                          "starting_stack": STARTING_STACK, "big_blind": BIG_BLIND,
                          "small_blind": 1, "stacks_reset_each_hand": True,
                          "flush_beats_full_house": True, "trips_beat_straight": True,
                          "lowest_straight": "A6789", "time_bank_enforced": False}}


class SessionStore:
    def __init__(self, hands=1000, bot=strategy, max_sessions=MAX_SESSIONS, ttl=SESSION_TTL_SECONDS):
        self.hands, self.bot, self.max_sessions, self.ttl = hands, bot, max_sessions, ttl
        self.lock = threading.RLock()
        self.sessions = OrderedDict()

    def _expire(self):
        cutoff = time.monotonic() - self.ttl
        for token in list(self.sessions):
            if self.sessions[token].touched < cutoff:
                del self.sessions[token]
        while len(self.sessions) >= self.max_sessions:
            self.sessions.popitem(last=False)

    def get(self, token=None, reset=False):
        with self.lock:
            # Expire before look-up; a stale browser cannot recover old memory.
            cutoff = time.monotonic() - self.ttl
            if token in self.sessions and self.sessions[token].touched < cutoff:
                del self.sessions[token]
            if reset:
                self.sessions.pop(token, None)
            if token not in self.sessions:
                self._expire()
                token = secrets.token_urlsafe(32)
                self.sessions[token] = HumanSession(self.hands, self.bot)
            session = self.sessions[token]
            session.touched = time.monotonic()
            self.sessions.move_to_end(token)
            return token, session


def create_app(hands_per_session=1000, bot=strategy):
    if not 1 <= hands_per_session <= 10000:
        raise ValueError("hands_per_session must be between 1 and 10000")
    application = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    application.state.sessions = SessionStore(hands_per_session, bot)

    @application.middleware("http")
    async def no_cache(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        return response

    def session_for(request, response, reset=False):
        token, session = application.state.sessions.get(request.cookies.get(COOKIE), reset)
        response.set_cookie(COOKIE, token, httponly=True, samesite="strict",
                            secure=request.url.scheme == "https", max_age=SESSION_TTL_SECONDS)
        return session

    @application.get("/")
    def index():
        if not FRONTEND.is_file():
            raise HTTPException(503, "The poker table is starting up. Please refresh shortly.")
        return FileResponse(FRONTEND)

    @application.get("/api/state")
    def get_state(request: Request, response: Response):
        session = session_for(request, response)
        with session.lock:
            return session.state()

    @application.post("/api/action")
    def action(body: ActionBody, request: Request, response: Response):
        session = session_for(request, response)
        with session.lock:
            return session.act(body.request_id, body.action)

    @application.post("/api/next-hand")
    def next_hand(body: NextBody, request: Request, response: Response):
        session = session_for(request, response)
        with session.lock:
            return session.next_hand(body.request_id)

    @application.post("/api/new-session")
    def new_session(body: dict, request: Request, response: Response):
        session = session_for(request, response, reset=True)
        with session.lock:
            return session.state()

    return application


app = create_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--hands", type=int, default=1000)
    arguments = parser.parse_args()
    uvicorn.run(create_app(arguments.hands), host=arguments.host, port=arguments.port,
                access_log=False)
