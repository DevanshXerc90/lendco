# API

The Vercel entry point is `api/index.py`. Local development may leave demo mode
open; configured or production deployments require `X-API-Key`.

| Method | Endpoint | Purpose | Auth |
|---|---|---|---|
| GET | `/api/health` | Health and mode | public |
| GET | `/api/borrowers` | List synthetic borrowers | demo key when configured |
| GET | `/api/borrowers/{id}` | Borrower profile | demo key when configured |
| GET | `/api/payments?borrower_id=` | Payment history | demo key when configured |
| GET | `/api/documents` | KB document list | demo key when configured |
| GET | `/api/eval` | Evaluation artifact | demo key when configured |
| POST | `/api/chat` | One deterministic chat turn | demo key |
| POST | `/api/scenario` | Run a scripted scenario | demo key |

Chat and scenario requests are limited to 16 KiB and 30 requests per IP per
minute per endpoint. The in-memory limiter is suitable for a demo, not a
multi-instance production deployment.
