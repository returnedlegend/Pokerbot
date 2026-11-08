"""Human exhibition protocols preserve the official information boundary."""
import time

from fastapi.testclient import TestClient

from game import Action, Deal, Game
from tools import human_play as human


class CallingBot:
    @staticmethod
    def new_half(config, memory):
        memory.clear()
        memory.update(observations=[], results=[])

    @staticmethod
    def choose_action(observation, memory):
        memory["observations"].append(observation)
        return Action.CHECK_CALL

    @staticmethod
    def on_hand_end(result, memory):
        memory["results"].append(result)


def client_for(bot=CallingBot, hands=5):
    return TestClient(human.create_app(hands, bot))


def play_call_hand(client, state=None):
    state = state or client.get("/api/state").json()
    for _ in range(25):
        if state["hand_over"]:
            return state
        assert "CHECK_CALL" in state["observation"]["legal_actions"]
        response = client.post("/api/action", json={"request_id": state["request_id"], "action": "CHECK_CALL"})
        assert response.status_code == 200, response.text
        state = response.json()
    raise AssertionError("A hand did not terminate within the legal raise cap")


def stored_session(client):
    return client.app.state.sessions.sessions[client.cookies.get(human.COOKIE)]


def test_initial_state_only_serializes_public_observation():
    client = client_for()
    response = client.get("/api/state")
    state = response.json()
    session = stored_session(client)
    assert response.headers["cache-control"] == "no-store"
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    assert state["observation"]["hole_cards"] == list(session.game.deal.hole_cards[0])
    assert len(state["observation"]["hole_text"]) == 5
    assert state["observation"]["board"] == []
    assert state["bot_cards"] is None and state["result"] is None
    assert state["observation"]["current_made_hand_category"] is None
    assert not {"deal", "seed", "memory", "bot_hole_cards"}.intersection(state)
    assert session._seed not in response.text
    before = session.state()
    # Equal visible cards/history must produce equal public data regardless of
    # the actual hidden cards. This checks serialization, not just key names.
    hidden = [card for card in range(36) if card not in session.game.deal.hole_cards[0]]
    alternate = Deal((session.game.deal.hole_cards[0], tuple(hidden[-5:])), tuple(hidden[:5]))
    session.game = Game(alternate, button_seat=0, hand_index=0)
    assert session.state() == before


def test_illegal_and_duplicate_action_do_not_mutate_a_second_time():
    client = client_for()
    first = client.get("/api/state").json()
    bad = client.post("/api/action", json={"request_id": first["request_id"], "action": "LOOK_AT_BOT_CARDS"})
    assert bad.status_code == 400
    assert client.post("/api/action", json={"request_id": "☃", "action": "CHECK_CALL"}).status_code == 409
    assert client.get("/api/state").json() == first
    response = client.post("/api/action", json={"request_id": first["request_id"], "action": "CHECK_CALL"})
    assert response.status_code == 200
    after = response.json()
    stale = client.post("/api/action", json={"request_id": first["request_id"], "action": "CHECK_CALL"})
    assert stale.status_code == 409
    assert client.get("/api/state").json() == after


def test_fold_does_not_reveal_private_cards_or_unseen_board():
    client = client_for()
    state = client.get("/api/state").json()
    terminal = client.post("/api/action", json={"request_id": state["request_id"], "action": "FOLD"}).json()
    assert terminal["hand_over"]
    assert terminal["bot_cards"] is None
    assert terminal["result"]["opp_hole"] is None
    assert terminal["observation"]["board"] == terminal["result"]["board"] == []
    assert terminal["result"]["bot_category"] is None
    assert terminal["result"]["human_net"] == -1
    assert terminal["score"]["human_chips"] == -terminal["score"]["bot_chips"]
    assert terminal["score"]["hands_played"] == 1


def test_showdown_reveals_legally_and_settles_exactly_once():
    client = client_for()
    terminal = play_call_hand(client)
    session = stored_session(client)
    assert terminal["result"]["went_to_showdown"]
    assert terminal["bot_cards"] == human.card_text(session.game.deal.hole_cards[1])
    assert terminal["result"]["opp_hole"] == list(session.game.deal.hole_cards[1])
    assert len(terminal["observation"]["board"]) == 5
    assert terminal["result"]["human_category"] and terminal["result"]["bot_category"]
    assert terminal["score"]["hands_played"] == 1
    for _ in range(3):
        assert client.get("/api/state").json() == terminal
    assert len(session.memory["results"]) == 1
    assert terminal["result"]["human_net"] + terminal["result"]["bot_net"] == 0


def test_next_hand_is_explicit_rotates_dealer_and_rejects_double_deal():
    client = client_for()
    first = client.get("/api/state").json()
    assert client.post("/api/next-hand", json={"request_id": first["request_id"]}).status_code == 409
    terminal = play_call_hand(client, first)
    next_response = client.post("/api/next-hand", json={"request_id": terminal["request_id"]})
    assert next_response.status_code == 200
    next_state = next_response.json()
    assert next_state["hand_number"] == 2
    assert next_state["observation"]["is_button"] != first["observation"]["is_button"]
    assert next_state["observation"]["my_stack"] == 78
    assert next_state["score"]["hands_played"] == 1
    assert client.post("/api/next-hand", json={"request_id": terminal["request_id"]}).status_code == 409
    assert client.get("/api/state").json() == next_state


def test_browser_sessions_and_reset_do_not_share_learning_or_scores():
    app = human.create_app(3, CallingBot)
    first, other = TestClient(app), TestClient(app)
    first_state = first.get("/api/state").json()
    other_state = other.get("/api/state").json()
    assert first.cookies.get(human.COOKIE) != other.cookies.get(human.COOKIE)
    first_session, other_session = stored_session(first), stored_session(other)
    assert first_session.memory is not other_session.memory
    assert first_session._seed != other_session._seed
    play_call_hand(first, first_state)
    assert other.get("/api/state").json() == other_state
    old_token = first.cookies.get(human.COOKIE)
    fresh = first.post("/api/new-session", json={}).json()
    assert first.cookies.get(human.COOKIE) != old_token
    assert old_token not in app.state.sessions.sessions
    assert fresh["score"]["hands_played"] == 0
    assert fresh["score"]["human_chips"] == 0
    assert stored_session(first).memory["results"] == []


def test_cross_site_form_cannot_reset_the_cookie_session():
    client = client_for()
    before = client.get("/api/state").json()
    token = client.cookies.get(human.COOKIE)
    response = client.post("/api/new-session", data={"reset": "1"}, headers={"Origin": "https://elsewhere.invalid"})
    assert response.status_code == 422
    assert client.cookies.get(human.COOKIE) == token
    assert client.get("/api/state").json() == before


def test_session_limit_and_expiration_bound_retained_private_state():
    store = human.SessionStore(2, CallingBot, max_sessions=2, ttl=60)
    a, a_session = store.get()
    b, _ = store.get()
    store.get(a)
    c, _ = store.get()
    assert len(store.sessions) == 2 and a in store.sessions and c in store.sessions and b not in store.sessions
    a_session.touched = time.monotonic() - 61
    replacement, _ = store.get(a)
    assert replacement != a and a not in store.sessions


def test_completed_session_requires_explicit_reset():
    client = client_for(hands=1)
    terminal = play_call_hand(client)
    assert terminal["session_complete"]
    assert client.post("/api/next-hand", json={"request_id": terminal["request_id"]}).status_code == 409
    assert client.post("/api/new-session", json={}).json()["hand_number"] == 1


def test_actual_policy_plays_legal_hands_and_zero_sum_results():
    client = client_for(bot=human.strategy, hands=3)
    state = client.get("/api/state").json()
    for index in range(3):
        terminal = play_call_hand(client, state)
        assert terminal["error"] is None
        assert terminal["score"]["human_chips"] + terminal["score"]["bot_chips"] == 0
        assert terminal["score"]["hands_played"] == index + 1
        if index < 2:
            state = client.post("/api/next-hand", json={"request_id": terminal["request_id"]}).json()


def test_callback_error_pauses_play_without_revealing_exception_details():
    class BrokenBot(CallingBot):
        @staticmethod
        def choose_action(observation, memory):
            raise RuntimeError("private diagnostic seed and cards must not reach the browser")

    client = client_for(BrokenBot)
    first = client.get("/api/state").json()
    response = client.post("/api/action", json={"request_id": first["request_id"], "action": "CHECK_CALL"})
    assert response.status_code == 200
    state = response.json()
    assert state["error"]
    assert "private diagnostic" not in response.text
    assert state["bot_cards"] is None and state["result"] is None
    assert client.post("/api/action", json={"request_id": state["request_id"], "action": "CHECK_CALL"}).status_code == 503
