# Demo script (about 5 minutes)

Run `streamlit run app.py` first and keep a terminal open for the test run at the end.

| Time | What to show | What to say |
|---|---|---|
| 0:00 | Title and README diagram | The problem: planning a trip means juggling flights, hotels, weather and costs. This assistant does it with an agent that decides which tools to use. |
| 0:30 | **Form tab**: Bangalore to Goa, 3 days, 1 traveler, budget 25000 | Click *Plan my trip*. Point at the live progress messages: the agent is choosing and calling tools. |
| 1:00 | Scroll the result | Flights, hotel, weather, places per day, budget breakdown, total vs budget, and **Why we chose this**. Download the JSON. |
| 1:45 | Expand **How the agent got here** | These are the tool calls the agent chose by itself: flights both ways, weather, hotels, places, budget. |
| 2:15 | Text tab: `Plan a 4 day trip from Hyderabad to Goa starting 12 December for 2 people, budget 40000` | Natural language works. Two travelers means one room. It is slightly over budget and says so. |
| 2:45 | Text tab: `Plan a 3 day trip from Delhi to Goa starting 15 December` | The dataset has no Delhi to Goa flight. The agent says so and still plans the hotel, places and weather. |
| 3:15 | Text tab: `Cheapest hotels in Goa with a pool` then `Where can I fly from Delhi?` | The same agent answers quick questions, with tables built from the data. |
| 3:30 | Text tab: `Compare hotels in Goa` then `Is the fastest flight from Bangalore to Goa worth the extra cost?` | The agent chooses a comparison tool and states the trade-off with real numbers. |
| 3:45 | Text tab: `Places to visit in Kolkata` | Every place shown comes from the dataset. Explain grounding: the model picks IDs, code rebuilds every fact and recomputes the budget, and invented places are removed. |
| 4:15 | Terminal: `python -m pytest -q` and `ruff check .` | 164 tests pass without a network or API key; lint is clean; code follows PEP 8 with docstrings. |
| 4:45 | README *Known limitations* | Sample data limits (25 of 56 routes, generic place names) and next steps. |

Checklist before recording: `.env` has a valid key, you have run each prompt once, and the browser window is wide
enough to show the day cards in one row.

Evaluation criteria and where to show them:

* **Problem and use case**: opening 30 seconds and the README.
* **Use of the datasets and weather API**: the result page (flights, hotels, places, live forecast) and the tool-call list.
* **Agentic workflow and LangChain**: *How the agent got here*, parallel tool calls, retries, intent routing.
* **Code quality and structure**: project tree, tests and lint run.
* **Final output quality**: the itinerary screen and the downloaded JSON.
