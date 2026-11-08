# Play against jr007

Start the local table from the repository root:

```sh
.venv/bin/python tools/human_play.py --port 8080
```

Open **http://127.0.0.1:8080** in a browser on the same machine. In this cloud
environment, open the **Desktop** view and use Chrome there. A localhost URL
on your own computer reaches your own computer, so run the command locally if
you want to use your own machine's browser instead.

The table runs the actual validated `strategy.py` policy with the unchanged
official game and evaluator. This is short-deck five-card Omaha: use exactly
two private cards and three community cards. Flushes beat full houses; trips
beat straights. The deck contains 6 through ace, with A6789 the lowest straight.

Use **Fold**, **Check/Call**, **Half pot**, or **Pot**. The buttons show the exact
legal chip amounts. After a hand, choose **Deal next hand**. Dealer position
alternates, and both players start every hand with 80 practice chips. Session
scores add up net chips across hands. There is no real money or thinking bank.

Each browser receives a separate session and private deal stream. The bot sees
only its legal cards, public board and action history. Its learning memory
persists within a session; **New session** clears it and the score. Bot cards
and unrevealed community cards remain hidden after a fold. Showdowns reveal
the bot's five cards. Sessions expire after six hours of inactivity.

The overnight search did not produce a candidate that passed every protected
release gate, so this table uses the preserved validated bot. No experimental
candidate is substituted for it.
