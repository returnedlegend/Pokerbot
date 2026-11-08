# Play against jr007

Start the local table from the repository root:

```sh
npm install
npm run dev
```

Open **http://127.0.0.1:3000** in a browser on the same machine. Node 20 or newer
is required. The npm launcher serves the table and forwards its API requests
to the actual Python bot. It reuses an existing jr007 backend or starts it
with the project's virtual environment. Set up Python and `requirements.txt`
as described in the repository README when running on a fresh machine.

In this cloud
environment, open the **Desktop** view and use Chrome there. A localhost URL
on your own computer reaches your own computer, so run the command locally if
you want to use your own machine's browser instead.

`npm start` runs the same launcher. Choose another frontend port with
`npm run dev -- --port 3001`. `POKER_BACKEND_PORT` selects the internal Python
port, and `POKER_PYTHON` can specify another configured Python interpreter.
The direct Python entry point remains `python tools/human_play.py --port 8080`.

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
