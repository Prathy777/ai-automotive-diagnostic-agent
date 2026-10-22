# Graph Report - Hacktoberfest  (2026-10-22)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 230 nodes · 603 edges · 11 communities (7 shown, 4 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 43 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2
- Community 3
- Community 4
- Community 5
- Community 6
- Community 10

## God Nodes (most connected - your core abstractions)
1. `GemmaProvider` - 39 edges
2. `investigate()` - 36 edges
3. `Vehicle` - 22 edges
4. `AgentDecision` - 20 edges
5. `GemmaProviderTests` - 20 edges
6. `ProviderRequestError` - 16 edges
7. `tool_descriptions()` - 16 edges
8. `ProviderConfigurationError` - 15 edges
9. `create_provider()` - 15 edges
10. `provider_health()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `investigate()` --uses--> `ProviderRequestError`  [INFERRED]
  backend/app/agent.py → backend/app/providers/base.py
- `chat()` --uses--> `ProviderRequestError`  [INFERRED]
  backend/app/main.py → backend/app/providers/base.py
- `GemmaProvider` --uses--> `ProviderRequestError`  [INFERRED]
  backend/app/providers/gemma.py → backend/app/providers/base.py
- `Handler` --uses--> `ProviderRequestError`  [INFERRED]
  backend/run_local.py → backend/app/providers/base.py
- `GemmaProvider` --uses--> `AgentDecision`  [INFERRED]
  backend/app/providers/gemma.py → backend/app/agent_schemas.py

## Import Cycles
- None detected.

## Communities (11 total, 4 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.07
Nodes (6): GemmaProvider, decision_dict(), GemmaProviderTests, fake_urlopen(), fake_urlopen(), response_for()

### Community 1 - "Community 1"
Cohesion: 0.10
Nodes (17): search_knowledge(), chat(), ChatRequest, config(), dashboard(), demo_enabled(), get_investigation(), health() (+9 more)

### Community 2 - "Community 2"
Cohesion: 0.15
Nodes (8): Vehicle, _read(), run(), _recent(), _test(), run(), Tool, Handler

### Community 3 - "Community 3"
Cohesion: 0.16
Nodes (11): ProviderRequestError, main(), tool_descriptions(), main(), _redact(), main(), load_dotenv(), main() (+3 more)

### Community 4 - "Community 4"
Cohesion: 0.23
Nodes (4): investigate(), AgentHarnessTests, decision(), StaticProvider

### Community 5 - "Community 5"
Cohesion: 0.36
Nodes (11): api(), escapeHtml(), investigate(), loadStatus(), root, runButton, scenarios, sendChat() (+3 more)

### Community 10 - "Community 10"
Cohesion: 0.09
Nodes (8): AgentDecision, Diagnosis, Hypothesis, ProviderConfigurationError, DemoDecisionProvider, create_provider(), AgentTests, NoFinishProvider

## Knowledge Gaps
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GemmaProvider` connect `Community 0` to `Community 10`, `Community 3`?**
  _High betweenness centrality (0.219) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `GemmaProvider` (e.g. with `AgentDecision` and `ProviderConfigurationError`) actually correct?**
  _`GemmaProvider` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.06821480406386067 - nodes in this community are weakly interconnected._
- **Why does `investigate()` connect `Community 4` to `Community 0`, `Community 1`, `Community 2`, `Community 3`, `Community 6`, `Community 10`?**
  _High betweenness centrality (0.154) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `investigate()` (e.g. with `AgentDecision` and `Diagnosis`) actually correct?**
  _`investigate()` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.09957325746799431 - nodes in this community are weakly interconnected._
- **Why does `AgentDecision` connect `Community 10` to `Community 0`, `Community 3`, `Community 4`, `Community 6`?**
  _High betweenness centrality (0.097) - this node is a cross-community bridge._