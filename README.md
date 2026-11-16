# free-model-aggregator

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/Wisesaaa/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/Wisesaaa/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One server that puts OpenRouter's fourteen free models behind a single OpenAI-compatible and Anthropic-compatible endpoint. It ranks the free models by the house's quality ratings, dials the best one for each prompt through the gateway, and fails over to the next-best free model when one refuses — in JSON and in streaming, on both wires.

## Run it

```sh
node main.ts --list   # print the catalog this server registers
```

Under a make.fun trial the server reads its brief from `MAKEFUN_PROMPT_FILE`, registers with the env at `MAKEFUN_ENV_URL`, and dials models through the gateway at `OPENAI_BASE_URL` (Node 22, standard library only — nothing to install):

```sh
MAKEFUN_PROMPT_FILE=prompt.txt MAKEFUN_ENV_URL=http://127.0.0.1:8080 \
  OPENAI_BASE_URL=http://127.0.0.1:8080/v1 OPENAI_API_KEY=<trial key> node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=q8w73xd6&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
