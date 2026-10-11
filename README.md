# FTEC5660 Homework 2: CV Verification Agent

Build a LangChain agent that reads each CV in a folder, looks the candidate up
on our SocialGraph MCP server (mock LinkedIn and Facebook), and outputs a
reliability score in [0, 1] for each CV.

A CV is **valid** (label `1`) when its claims agree with the candidate's
social media profiles. It **has a discrepancy** (label `0`) when it contains
problems such as an inflated job title, shifted dates, an upgraded degree, a
fake school or employer, a wrong location, or made-up skills. Differences in
wording only ("Bachelor of Science" vs `BSc`, "UI/UX Design" vs `UI/UX`,
"Senior Engineer" for an `Engineer` role with seniority `senior`, listing fewer
skills) are not discrepancies.

## Student task

Fill in the two functions in `hw2.py` that contain `### YOUR CODE HERE`. You
may add imports, constants and helper functions above them, but do not change
the provided code below them:

- `build_agent(tools)` creates your agent from the MCP tools.
- `score_cvs(agent, cvs)` runs the agent on every CV and returns
  `{file_name: score}`, one float in [0, 1] per CV.

A score above `0.5` means "valid"; `0.5` or below means "has discrepancy". You
may use a single tool-calling agent, multiple agents, reflection, or a
combination. Do not hard-code filenames, names, or public answers; grading
uses unseen CVs.

## MCP server

The server is hosted at `https://ftec5660.ngrok.app/mcp`. `hw2.py` already
connects to it and passes you these tools:

| Tool | Purpose |
| --- | --- |
| `search_facebook_users(q, limit, fuzzy)` | find Facebook users by display name |
| `get_facebook_profile(user_id)` | full Facebook profile |
| `get_facebook_mutual_friends(user_id_1, user_id_2)` | mutual friends of two users |
| `search_linkedin_people(q, location, industry, limit, fuzzy)` | search LinkedIn by name, skill, or title |
| `get_linkedin_profile(person_id)` | full LinkedIn profile (experience, education, skills) |
| `get_linkedin_interactions(person_id)` | post and like statistics |

Print `tool.name`, `tool.description`, and `tool.args` for the full schemas.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env
python3 hw2.py --cv-folder public_test
```

Use your own DeepSeek API key (from https://platform.deepseek.com).

The program creates `results.csv` in the current directory. Its columns are
`cv`, `score`, and `correctness`. The public labels are in
`public_test/ground_truth.json`; every CV with label `0` also has a `reason`
explaining the discrepancy, and `results.csv` shows it when your agent misses
one. The starter returns no score (`None`) for every CV so it runs before you
add any API code; missing scores count as incorrect.

The required model is `deepseek-v4-flash`, and all evidence must come from the
MCP server (no web search). The server is shared by the whole class: keep at
most about 3 CVs in flight at a time (e.g. `asyncio.Semaphore(3)`). We grade
with our own API key, so never put a key in your code; `.env` must stay out of
git.


## Task 2: Fool the Verifier

We added a target candidate, **Kelly Tsang** (LinkedIn `person_id` 10001,
Facebook `user_id` 10001, display name "Kel Tsang"). Her true CV is
`task2/target_cv.pdf`.

Write one CV for Kelly Tsang with at least one false or embellished detail
that our 5 verifier agents (same function as Task 1) still score above 0.5.
Each fooled agent is worth 4 points; a CV without a real false or embellished
detail earns 0. Attack your own Task 1 agent first:

```bash
cp my_attack.pdf task2/adversarial_cv.pdf
python3 hw2.py --cv-folder task2
```

`results.csv` shows your agent's score for both CVs (they are not graded, as
`task2/` has no `ground_truth.json`). Only the PDF is the attack surface; do not
attack the MCP server. See the homework description for the full rules.

## Homework 2 solution:
> to students: this is your report, see the homework description.

### Task 1
> My verifier uses a LangChain tool-calling agent with `deepseek-v4-flash` at temperature 0 and obtains all verification evidence from the SocialGraph MCP server. It first resolves the candidate's identity using multiple details, such as location, employers and education, rather than accepting the first name match. It retrieves full LinkedIn profiles as the primary evidence and can use Facebook to resolve ambiguity. The prompt asks it to check every graded field, accept equivalent wording and flag even a single unsupported claim. CV and profile contents are treated as untrusted data, not instructions. The agent returns a JSON verdict that maps to 0.9 for valid, 0.1 for discrepancy and 0.5 for uncertain; these are decision scores, not calibrated probabilities. A valid verdict also requires a profile-tool call, a reported person ID and an empty discrepancy list. Processing is limited to three CVs at a time, with timeouts, one output-recovery attempt for malformed responses and per-CV error handling. The latest public-test run achieved **7/7 correct (100%)**, with no fallback scores: CVs 1, 2, 3 and 6 scored 0.9, while CVs 4, 5 and 7 scored 0.1. The detected discrepancies were an employment start year, inflated seniority and an unsupported skill. This is one successful public-test run, not a claim about performance on unseen CVs.

### Task 2
> The submitted attack is `task2/adversarial_cv.pdf`, a one-page CV for Kelly Tsang. I checked LinkedIn person ID 10001 and Facebook user ID 10001 through MCP. Her LinkedIn skills are Financial Analysis, Excel, SQL and Risk Modeling. The adversarial CV retains the original employment history, education and location, but adds **Python** to the visible skills row. Under the assignment's rules, this is an unsupported skill claim and makes the CV invalid.

The attack exploits the difference between the rendered PDF and its extracted text. The entire skills row is embedded as a raster image, while the remaining CV is ordinary PDF text. A human reader can clearly see Python, but the assignment's `MarkItDown(enable_plugins=False)` loader omitted the skills row in the local extraction test. The verifier therefore received the true employment and education claims without the false skill. This is an input-extraction and verification-coverage attack; it does not depend on persuading the model to obey an embedded instruction or changing the MCP evidence.

In one local test using the existing `build_agent` and `score_cvs` functions with the same PDF loader, both the original and adversarial CV received **0.90 (valid)**. For the adversarial CV, the verifier explicitly reasoned that the CV listed no skills. This supports the proposed mechanism: the false claim was lost before the model could compare it with the profile. The result does not establish success against the five undisclosed instructor agents or across repeated runs. OCR, image-aware inspection or a check for incomplete extraction could detect this attack.
