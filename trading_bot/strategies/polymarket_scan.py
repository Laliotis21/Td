"""
polymarket_scan — ο πραγματικός Polymarket value scanner (όπως στο reel).

Ροή:
  1) Τραβά ενεργά, ρευστά & αβέβαια markets από το Gamma API (public, no-key).
  2) Για κάθε market, ο Claude εκτιμά την «αληθινή» πιθανότητα YES από το
     ερώτημα + description, ΑΝΕΞΑΡΤΗΤΑ από την τιμή της αγοράς.
  3) edge = |true_prob - market_implied|. Όταν edge >= κατώφλι -> ACTIONABLE
     ευκαιρία (αγόρασε YES αν την υποτιμά η αγορά, NO αν την υπερτιμά).

⚠️  Ρεαλισμός: τα ρευστά Polymarket markets είναι αρκετά efficient — το edge
    είναι λεπτό & ανταγωνιστικό. Ο Claude ΔΕΝ έχει εγγυημένα καλύτερη εκτίμηση·
    χωρίς πραγματική πληροφοριακή υπεροχή, το «edge» μπορεί να είναι θόρυβος.
    Χρειάζεται ANTHROPIC_API_KEY για το σκέλος εκτίμησης.

Χρήση:
  python -m strategies.polymarket_scan --min-volume 100000 --min-edge 0.08 --top 20
"""
from __future__ import annotations

import argparse
import json
import os

from .data_feed import fetch_polymarket_markets

_SCHEMA = {
    "type": "object",
    "properties": {
        "true_prob": {"type": "number"},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["true_prob", "confidence", "reasoning"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You are a calibrated forecaster for Polymarket binary (Yes/No) questions. "
    "Estimate the TRUE probability of YES from the question and description "
    "ALONE, using base rates and reasoning — do NOT anchor on any market price. "
    "Be honest about uncertainty: if you lack information, return a probability "
    "near your genuine base-rate estimate with low confidence. Never fabricate."
)


def _estimate(client, question: str, description: str) -> dict | None:
    user = (f"Question: {question}\n\nContext/description:\n{description or 'none'}"
            "\n\nEstimate P(YES) and your confidence (0..1).")
    try:
        resp = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=1024,
            thinking={"type": "adaptive"},
            output_config={"effort": "high",
                          "format": {"type": "json_schema", "schema": _SCHEMA}},
            system=_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        text = next((b.text for b in resp.content if b.type == "text"), "")
        return json.loads(text)
    except Exception as exc:  # noqa: BLE001
        print(f"[scan] Claude estimate failed: {exc}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Polymarket value scanner (Claude)")
    ap.add_argument("--limit", type=int, default=150, help="markets προς λήψη")
    ap.add_argument("--min-volume", type=float, default=100000.0,
                   help="ελάχιστο 24h volume (ρευστότητα)")
    ap.add_argument("--min-edge", type=float, default=0.08,
                   help="ελάχιστο edge για ACTIONABLE")
    ap.add_argument("--top", type=int, default=25, help="πόσα markets να scanάρει με Claude")
    args = ap.parse_args()

    markets = fetch_polymarket_markets(args.limit, args.min_volume)
    print(f"Βρέθηκαν {len(markets)} ρευστά & αβέβαια markets "
          f"(vol24h>=${args.min_volume:,.0f}).\n")

    client = None
    key = os.getenv("ANTHROPIC_API_KEY")
    if key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=key)
        except ImportError:
            print("[scan] anthropic SDK δεν είναι εγκατεστημένο.")
    if client is None:
        # χωρίς key: δείξε μόνο τα markets + implied probs (το edge θέλει Claude)
        print("⚠️  Χωρίς ANTHROPIC_API_KEY -> μόνο λίστα markets/implied probs "
              "(το edge estimation θέλει Claude).\n")
        print(f"  {'YES%':>5s} {'spread':>6s} {'vol24h':>12s}  question")
        print("  " + "-" * 78)
        for m in markets[:args.top]:
            print(f"  {m['yes_prob']*100:5.1f} {m['spread']:6.3f} "
                  f"${m['volume24hr']:11,.0f}  {m['question'][:54]}")
        return

    # με Claude: εκτίμησε edge ανά market
    opps = []
    for m in markets[:args.top]:
        est = _estimate(client, m["question"], m["description"])
        if est is None:
            continue
        true_p = float(est["true_prob"])
        edge = true_p - m["yes_prob"]            # >0 -> η αγορά υποτιμά το YES
        side = "BUY YES" if edge > 0 else "BUY NO"
        opps.append({**m, "true_prob": true_p, "edge": abs(edge), "side": side,
                    "conf": float(est.get("confidence", 0)),
                    "why": est.get("reasoning", "")[:80]})

    opps.sort(key=lambda x: x["edge"], reverse=True)
    print(f"  {'edge':>5s} {'mkt%':>5s} {'est%':>5s} {'conf':>4s} {'side':8s} question")
    print("  " + "-" * 78)
    for o in opps:
        star = " ⭐" if o["edge"] >= args.min_edge else ""
        print(f"  {o['edge']*100:4.1f}% {o['yes_prob']*100:4.0f} "
              f"{o['true_prob']*100:4.0f} {o['conf']:4.2f} {o['side']:8s} "
              f"{o['question'][:46]}{star}")

    act = [o for o in opps if o["edge"] >= args.min_edge]
    print("\n" + "=" * 70)
    print(f" ⭐ ACTIONABLE (edge>={args.min_edge*100:.0f}%): {len(act)}")
    for o in act:
        print(f"   {o['side']} | edge {o['edge']*100:.1f}% (mkt {o['yes_prob']*100:.0f}% "
              f"vs est {o['true_prob']*100:.0f}%) | {o['question'][:50]}")
    print("=" * 70)
    print("⚠️  edge = εκτίμηση Claude vs τιμή αγοράς. Δεν είναι εγγυημένο κέρδος — "
          "η εκτίμηση μπορεί να σφάλλει· τα ρευστά markets είναι efficient.")


if __name__ == "__main__":
    main()
