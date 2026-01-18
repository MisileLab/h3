<!-- Parent: ../AGENTS.md -->

# SERVER TESTS KNOWLEDGE BASE

## OVERVIEW
pytest unit tests for protocol schemas and usage metering.

## WHERE TO LOOK
| Task | Location | Notes |
| --- | --- | --- |
| Protocol tests | server/tests/test_protocol.py | Pydantic schema validation |
| Meter tests | server/tests/test_meter.py | UsageMeter behavior |

## CONVENTIONS
- Class-based tests (TestProtocolMessages, TestUsageMeter).
- Tests reset UsageMeter._daily_usage directly.

## ANTI-PATTERNS
- Do not change protocol tests without updating protocol.py + extension types.
