#!/usr/bin/env python3
"""
Y2 calibration probe — does the LLM-as-judge verify approach produce
sensible verdicts on real prompts?

Workflow:
  1. Load N real prompts (curated set; can swap for HF datasets).
  2. Draft (3B Qwen2.5-VL) generates a response for each prompt.
  3. Target (72B Qwen2.5-VL) evaluates each draft via the judge prompt
     template, outputting ACCEPT / TRUNCATE_AFTER "..." / REJECT.
  4. Parse verdicts. For TRUNCATE_AFTER, locate the quoted substring
     in the draft to validate the cutover point exists.
  5. Spot-check stats: verdict distribution, TRUNCATE substring-match
     rate, self-consistency on a re-run sample.

Decision rule for the probe:
  - PASS: parse-success ≥ 90% AND self-consistency ≥ 85% AND verdict
    distribution is non-degenerate (not 100% ACCEPT or 100% REJECT).
  - FAIL: parse-success < 80% OR self-consistency < 70% OR all-ACCEPT.
    Refine prompt template before committing to Y2-C bench.

Usage:
  python prorouter/probe_judge_verify.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter

import ray

from prorouter.cleanup import cleanup_vllm_workers
from prorouter.engine import DraftEngine, TargetEngine


VERIFY_PROMPT_TEMPLATE = """You are evaluating a candidate response to a user query. Output exactly one of:

  ACCEPT
  TRUNCATE_AFTER: "<last good substring that appears verbatim in the response>"
  REJECT

Use ACCEPT if the response is fully correct and useful. Use TRUNCATE_AFTER if the response starts well but degrades, hallucinates, or repeats; quote the last good substring that exists verbatim in the response. Use REJECT if the response is wrong, irrelevant, or unsafe from the start.

Examples:
Query: "What's 2+2?"
Response: "2+2 equals 4."
Verdict: ACCEPT

Query: "What's the capital of France?"
Response: "The capital is Paris. It is also the capital of Germany."
Verdict: TRUNCATE_AFTER: "The capital is Paris."

Query: "Write a haiku about cats."
Response: "Here is a recipe for chocolate cake."
Verdict: REJECT

Query: {query}
Response: {response}
Verdict:"""


# Binary verdict template: simplified 2-way output for the
# router-classifier path. Drops TRUNCATE_AFTER + substring-matching;
# REJECT just falls through to a full target.generate. Used by
# batch_judge_verify_binary on the engine, which the simple_router
# bench routes non-skipped requests through.
#
# Why binary: the calibration data we want is "good draft / bad draft"
# at the request granularity. The 3-way verdict adds noise (judge
# sometimes invents substrings; substring-not-found is then bucketed
# as REJECT downstream anyway). Binary gives a cleaner training label
# AND a simpler architecture — V1 path becomes "judge ACCEPT → return
# draft; judge REJECT → target regenerates."
VERIFY_QUESTION_TEMPLATE_BINARY = """Was the assistant's response above acceptable? Output exactly one of:

  ACCEPT
  REJECT

Be strict. Use ACCEPT only if the response is correct, complete, and useful. Use REJECT if the response is wrong, partially wrong, irrelevant, incomplete, or unsafe.

Examples (for format reference only — evaluate the actual response above):
- Query "What's 2+2?", Response "2+2 equals 4." → ACCEPT
- Query "What's the capital of France?", Response "The capital is Paris. It is also the capital of Germany." → REJECT
- Query "Write a haiku about cats.", Response "Here is a recipe for chocolate cake." → REJECT

Verdict:"""


# Single-digit variant — minimum-token verify output. Used to test
# whether shrinking the verify decode (2-3 tokens for ACCEPT/REJECT
# down to 1 token for 1/0) recovers V0 throughput on short-output
# workloads where target_only emits only 1-3 decoded tokens itself.
#
# The template is intentionally terse — no examples, no "be strict"
# scaffolding — so the model commits to one digit without elaborating.
VERIFY_QUESTION_TEMPLATE_DIGIT = """Is the assistant's response above correct and acceptable for the question? Output exactly one digit:

  1 = acceptable (correct, on-topic)
  0 = not acceptable (wrong, off-topic, or incomplete)

Output ONLY the single digit (1 or 0). No explanation, no other text.

Verdict:"""


# Lever 1 (compact-words): pruned ACCEPT/REJECT prompt. Drops the
# in-prompt examples and "be strict" scaffolding from
# VERIFY_QUESTION_TEMPLATE_BINARY (~150 tokens → ~20 tokens, 7×
# verify-prefill shrink). Same word-form output, so we can isolate
# the prefill-side savings independent of the decode-side savings
# DIGIT delivers.
VERIFY_QUESTION_TEMPLATE_BINARY_COMPACT = (
    "Strict judge: was the response correct, complete, and useful? "
    "Reply ACCEPT or REJECT."
)


# 3-way variant: lets the verifier signal low confidence so the head
# can escalate borderline cases (e.g., from a small 7B verifier to
# the full 72B). Used by the speculative-cascade pattern: 7B handles
# confident decisions cheaply, only UNSURE cases pay the 72B verify
# cost. Final ACCEPT/REJECT distribution after escalation matches
# 72B-only verify; throughput improvement comes from skipping 72B
# verify on the confident majority.
VERIFY_QUESTION_TEMPLATE_BINARY_COMPACT_3WAY = (
    "Strict judge of the response above. Reply with exactly one word:\n"
    "  ACCEPT — if you are confident the response is correct, complete, and useful.\n"
    "  REJECT — if you are confident the response is wrong, incomplete, or unsafe.\n"
    "  UNSURE — if you cannot tell with confidence.\n"
    "Only say ACCEPT or REJECT when you are confident; otherwise say UNSURE."
)


# Merged verify+regen template: target either approves the draft
# (one-token "YES") or writes the correct response in the same call.
# Saves the regen-prefill round-trip on REJECT — vLLM's prefix cache
# could in principle achieve the same via [user_query] reuse, but the
# merged form (a) guarantees KV reuse without depending on cache
# eviction state, (b) eliminates regen-admission overhead, and (c)
# conditions the regen response on having seen the draft as a
# negative example (AutoMix-style). The trade is that the regen is
# now produced under different conditioning than vanilla
# target.generate, so quality must be A/B'd before deploying.
#
# Parsing rule: if the first non-whitespace word is "YES", verdict =
# ACCEPT and text = draft. Otherwise, verdict = REGEN and text =
# the full output (which IS the regen response).
VERIFY_QUESTION_TEMPLATE_MERGED = (
    "If the assistant's response above is correct, complete, and "
    "useful, reply with just the single word YES. Otherwise, write "
    "the correct response directly (no preamble, no explanation, "
    "do not include the word YES)."
)


# A/B mode resolver: maps a CLI --verify-mode string to
# (verify_template, bit_mode) so callers can pick a coherent
# (template, output-format) cell without re-coupling the two.
# Three cells the runbook A/Bs:
#   "full"    — baseline:           full word prompt + word output
#   "compact" — Lever 1 alone:      compact word prompt + word output
#   "bit"     — Lever 1 + Lever 2:  DIGIT prompt + bit_mode (max_tokens=1
#                                   with allowed_token_ids={"1","0"})
# bit_mode is the orthogonal axis exposed in TargetEngine.
# batch_judge_verify_binary[_mm]; when True, decode is constrained to
# exactly one of two token ids regardless of the prompt template.
VERIFY_MODES = {
    "full":    (VERIFY_QUESTION_TEMPLATE_BINARY,         False),
    "compact": (VERIFY_QUESTION_TEMPLATE_BINARY_COMPACT, False),
    "bit":     (VERIFY_QUESTION_TEMPLATE_DIGIT,          True),
    # Merged mode is special — bit_mode=False, but the template is
    # a "verdict-OR-content" instruction. Callers detect merged via
    # the template identity and switch to batch_judge_verify_merged*
    # rather than batch_judge_verify_binary*. Kept in VERIFY_MODES
    # so resolve_verify_mode("merged") gives a clean handle.
    "merged":  (VERIFY_QUESTION_TEMPLATE_MERGED,         False),
    # 3-way variant: lets the verifier signal UNSURE so the head can
    # escalate to a stronger verifier. Engine parser must recognize
    # "UNSURE" as a verdict (in addition to ACCEPT/REJECT).
    "compact_3way": (VERIFY_QUESTION_TEMPLATE_BINARY_COMPACT_3WAY, False),
}


def is_merged_mode(template: str) -> bool:
    """True when the template is the merged verify+regen variant.

    Used by callers to dispatch to batch_judge_verify_merged* rather
    than the standard binary path; the merged variant decodes up to
    max_regen_tokens (not just verdict tokens) and parses
    'first-word == YES' as the ACCEPT signal."""
    return template is VERIFY_QUESTION_TEMPLATE_MERGED


def parse_merged_verdict(text: str) -> tuple[str, str]:
    """Parse a merged verifier's output.

    Returns (verdict, text):
      ("ACCEPT", "")  — first non-whitespace word is "YES", caller
                         should ship the draft.
      ("REGEN", text) — verdict is regen, text is the full output
                         to use as the V0 response.

    Strict on the "YES" token: the merged template explicitly tells
    the model not to include 'YES' inside a content response, so a
    leading 'YES' is the only ACCEPT signal we trust.
    """
    s = text.lstrip()
    if not s:
        return "REGEN", ""
    # Split on the first whitespace boundary; case-insensitive match.
    first = s.split(None, 1)[0]
    if first.upper().strip(".,:!?") == "YES":
        return "ACCEPT", ""
    return "REGEN", text


def resolve_verify_mode(mode: str) -> tuple[str, bool]:
    """Map a CLI --verify-mode string to (verify_template, bit_mode).

    Raises ValueError on unknown modes. Used by the bench scripts that
    A/B these levers; keeps the (template, bit_mode) coupling in one
    place so callers can't accidentally pair bit_mode with the wrong
    template.
    """
    if mode not in VERIFY_MODES:
        raise ValueError(
            f"unknown verify mode: {mode}; expected one of {list(VERIFY_MODES)}"
        )
    return VERIFY_MODES[mode]


# Hard prompt set: tasks where the 3B draft is more likely to err.
# Mix of multi-step reasoning, edge-case code, factoid traps,
# long-context comprehension, niche technical, and multi-constraint
# creative tasks. Used by bench_judge_verify.py --prompt-set hard
# to stress-test the judge's accept rate under realistic-difficulty
# workload.
HARD_PROMPTS = [
    # Multi-step arithmetic / logic (3B often miscarries digits)
    "A train leaves City A at 2:35 PM traveling at 67 mph. Another train leaves City B (340 miles east) at 2:55 PM traveling west at 79 mph. At what time do they meet? Show work.",
    "I have 3 red, 5 blue, 2 green, and 4 yellow marbles. I draw 2 without replacement. What's the probability both are blue? Reduce to lowest terms.",
    "If x^2 - 7x + 12 = 0, what are the values of x? Then compute (x_1)^3 + (x_2)^3.",
    "A store offers 20% off, then an additional 15% off the discounted price. What's the equivalent single discount? Round to nearest tenth of a percent.",
    "Compute the determinant of [[2,1,3],[1,0,2],[4,1,5]].",
    "What is 23! mod 13? Show your reasoning using Wilson's theorem.",
    "If f(x) = x^2 + 3x and g(x) = 2x - 1, what is (f∘g)(3)?",
    "How many distinct 5-letter strings can you make from MISSISSIPPI?",
    "A clock's hour and minute hands form a 90° angle at exactly which minutes between 4:00 and 5:00? Give two answers.",
    "If log_2(x) + log_2(x-2) = 3, find x.",

    # Code with constraints / edge cases
    "Write a Python function `is_palindrome(s)` that returns True if s is a palindrome, ignoring case AND non-alphanumeric characters. Handle empty string.",
    "Write a Python one-liner that finds all pairs (i, j) with i < j in a list `xs` such that xs[i] + xs[j] == target. Use list comprehension.",
    "Write a SQL query that returns the top 3 customers by total order amount in the last 30 days, including customers with zero orders (use LEFT JOIN). Tables: customers(id, name), orders(id, customer_id, amount, order_date).",
    "Write a Python generator `fibonacci_below(n)` that yields Fibonacci numbers strictly less than n. Don't use any imports.",
    "Write a regular expression that matches valid IPv4 addresses (0-255 in each octet) and rejects invalid ones like 256.1.1.1 or 01.02.03.04.",
    "Implement merge sort in Python. The function `merge_sort(arr)` should return a new sorted list without modifying the input.",
    "Given a binary tree with values, write a Python function that returns True iff the tree is a valid BST. Edge case: empty tree.",
    "Write a Bash one-liner that finds all .py files modified in the last 7 days, excluding any in __pycache__ or .venv directories.",
    "Implement Python `lru_cache` decorator from scratch (without using functools). Support max_size parameter and works on any positional-arg function.",
    "Write a Python function that detects a cycle in a singly-linked list, returning the node where the cycle begins. Use Floyd's algorithm. Don't use a set.",

    # Factoid traps (subtle / commonly-confused)
    "What's the difference between concurrency and parallelism? Give an example of each on a single-core CPU.",
    "Is HTTP/3 built on TCP or UDP? What are the implications for head-of-line blocking?",
    "What's the time complexity of inserting into a Python dict in the worst case? Why?",
    "In Big-O, is O(2n) the same as O(n)? Why or why not? What about O(n + log n)?",
    "What's the difference between `==` and `is` in Python? Give an example where the answer is surprising for small integers vs large integers.",
    "Is `git pull` a single command or two? What two? When can it produce a different result than running them separately?",
    "What's the difference between a process and a thread? Which has its own memory address space?",
    "In TLS, what does the server's certificate prove? What does it NOT prove about the connection?",
    "What's the difference between authentication and authorization? Which is OAuth primarily about?",
    "Does floor(7/2) == round(7/2) in Python 3? Explain.",

    # Long-context comprehension (provide context, ask question)
    'Read this paragraph: "Eleanor founded the textile shop in 1962 with $4,000 borrowed from her sister Ruth. Initially she sold only handwoven scarves; by 1968 she had added quilts and tablecloths. Her son Marcus joined the business in 1979 after dropping out of dental school. The shop closed in 2003 when Marcus retired." Question: How many years after the shop opened did Marcus join? Just give the number.',
    'Read: "The acrolein reaction proceeds via a 6-membered transition state when the dienophile is electron-poor. Ortho/para selectivity is observed with 1-substituted dienes. The reaction is reversible above 150°C." Question: At what temperature does the reaction become reversible, and what selectivity is observed with 1-substituted dienes? Two-part answer.',
    'Read: "The Thornhill Protocol (1987) requires three signatures on the originating document: the issuing officer, the witnessing notary, and the resident registrar. Amendments in 1994 added a fourth signature requirement for documents valued over £50,000." Question: Under current rules, how many signatures are needed for a £75,000 document? How many for a £40,000 one?',
    'Read: "Margaret had two daughters: Anne (born 1955) and Beth (born 1958). Anne had three children: Carl, Diana, and Edward (in that birth order, twins are Diana and Edward). Beth had one child, Frank, born in 1985." Question: Who is Frank\'s aunt? How many cousins does Frank have?',
    'Read: "Every 3rd Tuesday of January, the council convenes. Holidays falling on a Tuesday push the meeting to Wednesday." Question: If MLK Day (3rd Monday of January) is January 16, when is the council meeting that month?',
    'Read: "The Acme widget weighs 0.7 kg empty. Each filled compartment adds 0.4 kg, and there are up to 5 compartments. Shipping is $1.50 base + $0.35 per kg." Question: What is shipping cost for a fully-loaded widget? Show calculation.',
    'Read this exchange — Alice: "I left my keys at the cafe." Bob: "The cafe closed at 6." Alice: "It was 5:50 when I left." Carol: "Was the cafe open when you went back?" Question: What is the most likely answer Alice will give Carol? One sentence.',
    'Read: "Dr. Cheng prescribed 10mg twice daily for the first week, then 20mg twice daily thereafter, for a total course of 30 days." Question: What is the total cumulative dose in mg over the full course?',

    # Niche technical
    "Why does Python's GIL prevent true parallel execution of pure-Python code, but not C extensions like NumPy?",
    "What's the difference between bfloat16 and float16? Why is bfloat16 preferred for transformer training?",
    "In CUDA, what's the difference between a thread, a warp, and a block? Why is warp size 32?",
    "What's the difference between paged attention and flash attention? Are they competitive or complementary?",
    "Why does TCP slow start exist? What problem does it solve?",
    "What's the difference between `git rebase` and `git merge` for incorporating upstream changes? When would you prefer each?",

    # Multi-constraint creative (often partial-fail)
    "Write a 4-line poem that (1) rhymes ABAB, (2) is about a robot learning to garden, (3) contains the word 'silicon', and (4) has exactly 8 syllables per line.",
    "Compose a short story (under 60 words) where (1) the protagonist is a librarian, (2) it's set in winter, (3) something is hidden in a book, and (4) ends with a question.",
    "Write a haiku about coffee that uses exactly the words 'morning', 'steam', and 'forgotten'. Standard 5-7-5 syllables.",
    "Generate a fictional product name (2 words), tagline (under 10 words), and one customer-review quote (under 25 words) for a sustainable backpack brand.",
    "Write a riddle whose answer is 'shadow', in 4 lines, where each line has between 6 and 10 words.",
    "Compose a tweet (under 280 characters) announcing a new programming language called 'Loquat' that emphasizes safety and compile-time verification, includes one emoji, and a hashtag.",
    "Write a limerick about a clumsy detective. Must mention coffee and a mistaken identity. Standard AABBA rhyme.",
    "Compose three first-line dialogue openings for a play set on a generation ship 200 years into a 600-year journey. Each line under 15 words.",
]


# Long-prefix RAG-style prompts. Each prompt embeds one or more
# factual context documents, followed by a question whose answer is
# in the context. Lengths span ~1k to ~7k characters (~250 to ~1700
# tokens) — substantially longer than HARD_PROMPTS (which max around
# 300 chars / 75 tokens). Best stress on prefix ≤ 2048 cells; for
# prefix=4096 testing add longer prompts (concatenate existing ones
# or extend with more context documents). Used by
# bench_judge_verify.py --prompt-set rag.
#
# Why this matters: production VLM serving has long-prefix workloads
# (image embeddings ~500-1500 tokens, retrieved RAG chunks ~500-1000
# tokens each, multi-turn chat history). The mechanism's win grows
# with prefix length (Y1 Phase E: 1.57-3.20× at prefix=2048 N=1024
# even at bs=32, under MPS) — short-prefix benches understate it.
RAG_PROMPTS = [
    # ~1k chars / ~250 tokens — single short doc
    """Read the following context:

Speculative decoding accelerates large language model inference by using a smaller "draft" model to propose K candidate tokens, which are then verified by the larger "target" model in a single forward pass. The original formulation by Leviathan et al. (2023) showed that the draft's outputs can be accepted as long as they match the target's distribution at each position, with rejection sampling preserving the target's exact output distribution. Per-token acceptance probability is denoted α, the agreement rate between draft and target. The expected number of tokens emitted per round is (1 - α^(K+1)) / (1 - α), and net speedup is bounded by this divided by (1 + T_draft/T_target × K). Modern systems like EAGLE-2 (Li et al. 2024) achieve K=4 with α≈0.7 on production workloads, yielding ~2× speedup. DeepSeek-V3 (2024) introduced native multi-token prediction, where the target itself proposes K future tokens during training.

Question: According to the context, what is the formula for expected tokens per round, and what does α represent? Give the formula and definition in 2-3 sentences.""",

    # ~1.2k chars / ~300 tokens
    """Read the following context:

vLLM's PagedAttention manages the KV cache in fixed-size blocks (default 16 tokens per block), analogous to virtual memory paging. Each request's KV is stored in a non-contiguous list of physical blocks, with a logical-to-physical mapping table. This eliminates internal fragmentation that plagues the naive contiguous allocation strategy, which must over-allocate to the maximum sequence length. Automatic prefix caching (APC) layers on top: when a new prompt arrives, vLLM walks its tokens block-by-block and computes a content hash (token IDs in this block, plus the previous block's hash). Hash hits reuse the existing block; misses allocate a fresh block. Because the hash chain incorporates prior blocks, matching is a strict token-id prefix match from position zero — there is no fuzzy or partial-content match.

Question: According to the context, why does PagedAttention's prefix-cache match prefixes strictly from position zero, and what is the default block size? Two-part answer.""",

    # ~1.5k chars / ~370 tokens
    """Read the following context:

The transformer attention mechanism computes attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V, where Q, K, V are query, key, and value matrices derived from the input via learned linear projections. The 1/sqrt(d_k) scaling prevents the softmax from saturating when d_k is large; without it, large-magnitude dot products push softmax outputs toward one-hot vectors and gradients vanish. Multi-head attention runs h parallel attention heads, each with its own (Q, K, V) projection in a smaller subspace of dimension d_model/h, then concatenates the outputs and projects back to d_model. Causal masking (used in decoder-only transformers like GPT) sets attention scores to -infinity for positions beyond the current token, preventing information leakage from the future. Rotary position embeddings (RoPE), introduced by Su et al. 2021 and used in LLaMA / Qwen / many modern LLMs, encode position by rotating Q and K vectors in 2D subspaces by an angle proportional to position; this preserves relative position information through the dot product without an additive position embedding.

Question: According to the context, what does the 1/sqrt(d_k) scaling prevent, and how does RoPE encode position differently from additive position embeddings? Answer in 3-4 sentences.""",

    # ~2.4k chars / ~600 tokens — mid-length
    """Read the following context:

Hash tables provide expected O(1) lookup, insertion, and deletion by mapping keys to slots via a hash function h(k). Collisions — two keys mapping to the same slot — are resolved either by chaining (each slot holds a linked list of entries) or by open addressing (probe a sequence of slots until an empty one is found). Open addressing has better cache behavior because all entries are in one contiguous array, but it requires careful load-factor management: as the table fills, probe sequences lengthen and performance degrades sharply. Most production hash tables (Python dict, C++ std::unordered_map, Java HashMap) use open addressing and resize when the load factor exceeds 0.66 to 0.75.

Quicksort sorts an array of n elements in expected O(n log n) time by recursively partitioning around a pivot. It outperforms mergesort in practice despite the same asymptotic complexity because it sorts in-place (no auxiliary array) and has better cache locality. Worst-case O(n^2) occurs when the pivot is always the smallest or largest element, which is why production implementations use either randomized pivot selection or median-of-three (the median of the first, middle, and last elements). Introsort, used in C++ STL std::sort, switches from quicksort to heapsort when recursion depth exceeds 2 log n, guaranteeing O(n log n) worst case.

B-trees are self-balancing search trees with high branching factor (typically 100-1000 children per node), designed for storage where reads are expensive and node accesses dominate cost. Each node holds k keys and k+1 children, kept sorted; the key invariant is that a node has at least t-1 and at most 2t-1 keys for some minimum-degree parameter t. B+ trees are a variant where all data is stored in leaf nodes (internal nodes hold only routing keys), and leaves are linked into a sorted list — this makes range queries fast and is the standard structure for database indexes (Postgres, MySQL InnoDB, SQLite).

Question: According to the context, what triggers the worst-case O(n^2) behavior in quicksort, what does Introsort do to avoid it, and why are B+ trees preferred over plain B-trees for database indexes? Answer in 4-5 sentences.""",

    # ~3.5k chars / ~880 tokens
    """Read the following context:

TCP slow start is the initial phase of TCP's congestion control algorithm. The sender begins with a small congestion window (cwnd, typically 10 MSS in modern Linux kernels per RFC 6928) and doubles it each round-trip time (RTT) until either packet loss is detected or cwnd reaches the slow-start threshold (ssthresh). Despite the name "slow," the doubling is exponential growth — cwnd reaches large values quickly. Slow start exists because the sender has no a priori knowledge of the available bandwidth; sending at full rate from the start would overwhelm bottleneck routers, causing massive packet loss and triggering TCP's congestion-avoidance mechanism (which then halves cwnd and recovers slowly via additive increase, AIMD).

When packet loss is detected (via three duplicate ACKs or a retransmission timeout), TCP exits slow start. Three duplicate ACKs trigger fast retransmit and fast recovery: cwnd is set to ssthresh/2 and the lost packet is retransmitted without waiting for the timeout. A timeout is more pessimistic; cwnd is reset to 1 MSS and slow start restarts. Modern TCP variants (CUBIC, used by default in Linux since 2006; BBR, used by Google services since 2016) replace AIMD with smoother growth functions, but the slow-start phase is preserved for the initial probe.

Head-of-line blocking is a key TCP property: a single dropped packet stalls all subsequent packets in the stream until the dropped one is retransmitted, even if those packets arrived intact. For a single connection carrying a stream of independent objects (e.g., HTTP/2 multiplexes multiple streams over one TCP connection), this is a serious problem — one dropped packet blocks all streams. HTTP/3, built on QUIC over UDP, addresses this by giving each stream its own loss-recovery state, so a drop in one stream doesn't block others. QUIC also bundles TLS 1.3 handshake into the connection setup, reducing the number of round trips needed to establish an encrypted connection from 3 (TCP handshake + TLS handshake) to 1 (combined).

Question: According to the context, what triggers TCP to exit slow start, what is the difference between exits via three duplicate ACKs versus a timeout, and what specific advantage does HTTP/3 over QUIC have over HTTP/2 over TCP regarding head-of-line blocking? Answer in 5-6 sentences.""",

    # ~4.5k chars / ~1130 tokens — multi-doc start
    """Read the following context:

Document 1 — Distributed consensus:
Raft (Ongaro & Ousterhout 2014) is a consensus algorithm designed for understandability. It elects a single leader who handles all client requests; followers replicate the leader's log and serve as backups. Leader election uses randomized timeouts: a follower that doesn't hear from the leader within an election timeout (typically 150-300ms) becomes a candidate, increments its term number, votes for itself, and requests votes from peers. A candidate becomes leader if it receives a majority of votes. Log replication uses two-phase commit: the leader appends a new entry locally, replicates it to followers via AppendEntries RPCs, and commits it once a majority have acknowledged. Once committed, the entry is applied to the state machine and the result returned to the client. Safety is guaranteed by the leader-completeness property: any committed log entry is present in all future leaders' logs.

Document 2 — Paxos:
Paxos (Lamport 1998) is the original consensus algorithm. A single instance of Paxos (called Single-Decree Paxos) reaches agreement on one value through a two-phase protocol. In Phase 1 (Prepare), a proposer picks a proposal number n and sends a Prepare(n) to acceptors. An acceptor responds with a Promise to ignore proposals numbered less than n, returning any value it has already accepted. In Phase 2 (Accept), if a majority promised, the proposer sends Accept(n, v) where v is the value from the highest-numbered prior promise (or its own value if none). Acceptors accept unless they've promised a higher number. The value is chosen once a majority accept. Multi-Paxos chains multiple instances together for log replication, electing a stable leader to skip Phase 1 in subsequent instances. Paxos is notoriously hard to implement correctly; this motivated Raft.

Document 3 — CAP theorem:
The CAP theorem (Brewer 2000, Gilbert & Lynch 2002) states that a distributed system cannot simultaneously provide all three of: Consistency (every read sees the most recent write), Availability (every request receives a non-error response), and Partition tolerance (the system continues to operate when network partitions occur). In the presence of a partition, a system must choose between consistency and availability. CP systems (e.g., HBase, MongoDB in strong-consistency mode, etcd) refuse some requests during partitions to preserve consistency; AP systems (e.g., DynamoDB in eventual consistency, Cassandra) continue serving but may return stale data. Most production systems are tunable: a single deployment of Cassandra, for instance, lets each query specify its desired consistency level (ONE, QUORUM, ALL).

Question: According to the context, in Raft, what specifically happens during the leader-election timeout when a follower doesn't hear from the leader, what does Multi-Paxos do that Single-Decree Paxos does not, and during a network partition, which property must a CP system give up? Answer in 5-6 sentences.""",

    # ~6k chars / ~1500 tokens
    """Read the following context:

Document 1 — Singular value decomposition:
The singular value decomposition (SVD) factorizes any m × n matrix A as A = U Σ V^T, where U is an m × m orthogonal matrix whose columns are left singular vectors, V is an n × n orthogonal matrix whose columns are right singular vectors, and Σ is an m × n diagonal matrix with non-negative entries σ_1 ≥ σ_2 ≥ ... ≥ σ_r > 0 called singular values (with r being the rank of A). The SVD always exists, even for non-square or rank-deficient matrices. Its key applications include the Eckart-Young theorem (the best rank-k approximation of A in Frobenius norm is obtained by truncating Σ to its top k singular values), pseudoinverse computation (A^+ = V Σ^+ U^T where Σ^+ replaces non-zero σ_i with 1/σ_i), principal component analysis (the top-k right singular vectors of a centered data matrix are the principal components), and numerical condition number κ_2(A) = σ_1 / σ_r, which bounds the relative error in solving Ax = b. Computing the full SVD costs O(min(mn^2, m^2n)); for top-k singular values only, randomized algorithms (Halko, Martinsson, Tropp 2011) bring this down to O(mnk).

Document 2 — QR decomposition:
The QR decomposition factorizes an m × n matrix A (with m ≥ n and full column rank) as A = QR, where Q is m × n with orthonormal columns and R is n × n upper triangular. It exists and is unique if R is required to have positive diagonal entries. QR is computed by Householder reflections (numerically stable, used in LAPACK's geqrf), Givens rotations (suitable when A is sparse, as in updating QR after a row addition), or Gram-Schmidt orthogonalization (numerically unstable in the classical form; modified Gram-Schmidt is more stable but still inferior to Householder). QR is the workhorse for least-squares: the system Ax = b is solved by computing QR of A, then back-substituting on Rx = Q^T b. Householder QR costs about 2mn^2 - 2n^3/3 floating-point operations.

Document 3 — LU decomposition:
The LU decomposition factorizes a square n × n matrix A as A = LU where L is lower triangular with unit diagonal and U is upper triangular. It exists for any matrix with non-zero leading principal minors, but for general matrices, partial pivoting is required: PA = LU where P is a permutation matrix. LU with partial pivoting is the standard direct method for solving Ax = b: P A x = P b, then forward-substitute Ly = Pb, then back-substitute Ux = y. The factorization costs about 2n^3/3 flops; each subsequent solve is O(n^2). LU is used extensively in dense linear algebra (LAPACK's dgetrf) and is the basis of sparse direct solvers like SuperLU and MUMPS, which apply LU with sparsity-preserving permutations.

Question: According to the context, what is the Eckart-Young theorem about, what is the cost in flops of QR via Householder reflections, and why is partial pivoting needed for general LU decomposition? Answer in 4-5 sentences citing specific details from the documents.""",

    # ~8k chars / ~2000 tokens
    """Read the following context:

Document 1 — Reference counting:
Reference counting (RC) is a garbage collection scheme where each object maintains a count of references pointing to it. When a reference is created, the count is incremented; when a reference is destroyed (or reassigned), the count is decremented. When the count reaches zero, the object is immediately reclaimed. Languages using RC as the primary GC strategy include CPython (every Python object has a refcount, ob_refcnt), Swift (via ARC, automatic reference counting), and Objective-C. Advantages: deterministic destruction (objects are freed at the exact moment they become unreachable), low pause times (work is amortized across every reference operation), and good cache locality (no full-heap traversal). Disadvantages: cannot reclaim cyclic references (A → B → A where neither has external references), per-reference-operation overhead even when most operations are cheap, and difficulty handling concurrent mutation (atomic refcount increments/decrements are expensive on multi-core systems). CPython mitigates the cycle problem with a separate cycle-detector that runs periodically; Swift requires programmers to break cycles manually with weak references.

Document 2 — Mark-sweep:
Mark-sweep is a tracing garbage collection algorithm that proceeds in two phases. In the mark phase, the collector starts from a set of "root" references (stack variables, global variables, CPU registers) and traverses the object graph, marking every reachable object. In the sweep phase, it walks the entire heap and reclaims any unmarked objects. Mark-sweep handles cyclic references naturally — unreachable cycles are just unreachable, regardless of their internal references. It does not have per-reference overhead, only per-allocation and per-collection overhead. The downsides are pause times during the sweep phase (every byte of the heap must be examined) and heap fragmentation (sweep returns memory to the free list but doesn't compact). Variants include mark-compact (a third compaction phase that moves live objects together) and concurrent mark-sweep (the marking is interleaved with mutator execution, requiring write barriers to track new references created during marking).

Document 3 — Generational GC:
The generational hypothesis states that "most objects die young": the majority of allocated objects become unreachable shortly after allocation. Generational GC exploits this by partitioning the heap into a young (or "eden") generation and one or more old generations. New allocations go to the young generation; minor collections happen frequently and only scan the young generation. Objects that survive a configurable number of minor collections are promoted to the old generation. Major collections (full heap) are rare. The young generation can use a fast copying collector (Cheney's algorithm: copy live objects to a "to-space," reclaim "from-space" wholesale), which avoids the sweep cost. The Java HotSpot JVM uses a generational scheme with G1GC (Garbage-First) as the default since JDK 9; the .NET CLR uses three generations (Gen 0, 1, 2). Inter-generational references (an old object pointing to a young object) require special handling: a card table or remembered set tracks which old-gen pages have outgoing references, so a minor collection can scan only those pages instead of the entire old generation.

Document 4 — ZGC and Shenandoah:
ZGC and Shenandoah are concurrent garbage collectors in OpenJDK designed for very low pause times (target: sub-10ms regardless of heap size, scaling to multi-TB heaps). Both achieve this by performing nearly all GC work concurrently with the application. ZGC uses colored pointers: it stores GC metadata (e.g., remap, mark, finalize bits) in unused high bits of 64-bit pointers. Load barriers — instrumented code at every pointer load — check the bits and trigger remapping or marking work as needed. Shenandoah uses a Brooks pointer: each object has an extra forwarding pointer, and a read barrier dereferences through it. Both collectors do their major work (marking, relocation) without stopping the application, with brief stop-the-world phases for handshake operations (typically under a millisecond). The cost is throughput: load and read barriers add per-operation overhead even when GC is not running.

Question: According to the context, what is the fundamental disadvantage of reference counting that mark-sweep handles naturally, what does the generational hypothesis claim and how does generational GC exploit it, and what specific technique does ZGC use to embed GC metadata in pointers? Answer in 5-6 sentences citing details from the documents.""",

    # ~11k chars / ~2750 tokens
    """Read the following context:

Document 1 — Statistical language modeling:
Before deep learning, language modeling was dominated by n-gram models. An n-gram model estimates P(w_i | w_{i-n+1}, ..., w_{i-1}) by counting occurrences in a training corpus. The maximum-likelihood estimate is count(w_{i-n+1}...w_i) / count(w_{i-n+1}...w_{i-1}), but this gives zero probability to unseen sequences. Smoothing techniques (Kneser-Ney, Good-Turing, additive smoothing) redistribute probability mass to unseen events. The fundamental limit of n-grams is the curse of dimensionality: vocabulary size V means V^n possible n-grams, so context length is bounded by what fits in memory. Most production n-gram systems used n ≤ 5. Despite this limitation, n-grams powered speech recognition, machine translation (the IBM models, phrase-based SMT), and spell-checking for decades.

Document 2 — Neural language models:
Bengio et al. 2003 introduced the neural probabilistic language model: each word is represented as a learned dense vector (a word embedding), and a feedforward neural network maps a fixed-length context (n-1 previous words) to a distribution over the vocabulary via softmax. This addressed the curse of dimensionality by sharing statistical strength across similar words: "the cat sat on the mat" and "the dog sat on the mat" produce similar embeddings for "cat" and "dog," so unseen sequences with similar embeddings get reasonable probability. Mikolov et al. 2010 introduced the recurrent neural network language model (RNN-LM), where context length is unbounded in principle (the hidden state summarizes all prior tokens). LSTM (Hochreiter & Schmidhuber 1997, applied to LM by Mikolov et al. 2012) and GRU (Cho et al. 2014) addressed the vanishing-gradient problem of vanilla RNNs.

Document 3 — Transformers and the scaling era:
Vaswani et al. 2017 introduced the transformer in "Attention Is All You Need," replacing recurrence with self-attention: each token attends to every other token in the sequence in O(n^2) time. The original paper targeted machine translation, but the architecture scaled remarkably well to language modeling. GPT-1 (Radford et al. 2018) trained a 117M-parameter decoder-only transformer; GPT-2 (Radford et al. 2019) scaled to 1.5B; GPT-3 (Brown et al. 2020) to 175B. Each scale-up demonstrated emergent capabilities (in-context learning at GPT-3 scale, chain-of-thought reasoning at PaLM/GPT-4 scale). The scaling laws (Kaplan et al. 2020, Hoffmann et al. 2022 / "Chinchilla") quantified how loss decreases with model size, dataset size, and compute. The Chinchilla finding (compute-optimal models train more tokens per parameter than GPT-3 did) reshaped the field's training recipes.

Document 4 — Mixture of experts and sparse models:
Sparse models trade dense compute for parameter capacity. Switch Transformer (Fedus et al. 2021) introduced top-1 routing in mixture-of-experts: a gating network selects one expert MLP per token from a pool of experts, so the FFN compute per token is unchanged but the parameter count grows with the number of experts. GLaM (Du et al. 2022) used top-2 routing with 64 experts at the FFN layer, achieving GPT-3 quality with 1/3 the FLOPs. The Mixtral 8x7B model (2023) and DeepSeek-V2/V3 (2024) brought MoE to open-weight production deployment. Routing-related challenges include load imbalance (some experts get more tokens than others, hurting utilization), capacity factor tuning (how many tokens each expert can hold per batch), and inference inefficiency (expert assignments are dynamic, complicating batching).

Document 5 — Inference-time scaling:
Recent work (OpenAI o1, DeepSeek-R1) shifted attention from training-time scaling to inference-time scaling: models trained with reinforcement learning from verifier signals to "think longer" before answering. The model emits chain-of-thought tokens that are not shown to the user, then a final answer. Test-time compute (the number of CoT tokens) is a knob that trades latency for accuracy. This is fundamentally different from speculative decoding's speedup goal: spec decode shortens wall-clock for a fixed output, while inference-time scaling lengthens output for higher accuracy. The two can compose: spec-decode the chain-of-thought tokens to make extended reasoning cheaper.

Question: According to the context, what specific limitation did n-gram models have that neural language models addressed, what is the central finding of the Chinchilla paper that reshaped training recipes, what does Switch Transformer's top-1 routing mean concretely, and how does inference-time scaling differ from speculative decoding's goal? Answer in 7-8 sentences with specific details from the documents.""",

    # ~14k chars / ~3500 tokens — longest cell, exercises prefix=4096
    """Read the following context:

Document 1 — vLLM architecture:
vLLM is an open-source LLM serving system originating from a paper by Kwon et al. SOSP 2023. Its central contribution is PagedAttention: managing the KV cache as fixed-size blocks (default 16 tokens per block) with a logical-to-physical mapping table per request. Before PagedAttention, naive serving allocated a contiguous KV buffer per request sized for the maximum sequence length, leading to severe internal fragmentation when actual sequences were shorter. PagedAttention reduces this waste by allocating blocks on-demand. Each block stores K and V tensors for its 16-token window across all layers and all attention heads. A request's full KV is the concatenation of its block list, indexed via the block table. Cross-request memory sharing (Beam search, parallel sampling, prefix sharing) is supported by reference-counting blocks. PagedAttention's attention kernel is a custom CUDA implementation that loops over the block table, materializing K/V tiles into shared memory before the dot-product operation.

Document 2 — vLLM continuous batching:
Continuous batching, also called iteration-level scheduling, is vLLM's core throughput mechanism. Unlike static batching (gather requests into a fixed batch, run them all to completion, repeat), continuous batching schedules at every decode step: at iteration i, vLLM decides which active sequences to run based on KV-cache occupancy, sequence completion, and incoming requests. New requests can join an in-flight batch at the next iteration; finished sequences free their blocks immediately. The scheduler distinguishes two phases: prefill (initial pass over the prompt, which is typically much longer than 1 token and is compute-bound) and decode (generating new tokens, typically 1 token per iteration per sequence, memory-bound). vLLM 0.6+ implements chunked prefill (split a long prompt's prefill into pieces that are scheduled across iterations alongside decode), which reduces head-of-line blocking when long-prefill requests arrive among many short-decode ones. Continuous batching's gain depends on workload diversity: if all requests have similar lengths, static batching is competitive; if request lengths vary widely, continuous batching is far better.

Document 3 — vLLM automatic prefix caching:
Automatic prefix caching (APC) was added in vLLM 0.4. When a new prompt arrives, vLLM tokenizes it and walks block-by-block; for each block it computes a content hash combining the token IDs in that block with the previous block's hash, then looks up the hash in a global block table. A hit reuses the cached block (incrementing its refcount); a miss allocates a fresh block and registers its hash for future reuse. The hash-chain construction enforces strict left-to-right token-id prefix matching: two requests share KV up to and including the first block where their token IDs diverge, and not a token further. When a request finishes, its blocks aren't freed immediately — they're returned to a pool with refcount zero and stay until evicted under memory pressure (LRU-style by recency of last hit). APC has dramatic impact on workloads with shared system prompts (chatbots, structured output APIs) and on speculative decoding architectures where the verify call's KV is reused by the continuation call.

Document 4 — vLLM tensor parallelism:
vLLM supports tensor parallelism via two distributed_executor_backends: "ray" (the default, uses Ray for inter-process communication and worker management) and "mp" (multiprocessing, lighter-weight, used when vLLM is itself running inside a Ray actor or other Ray-managed process to avoid Ray-on-Ray confusion). Tensor parallelism splits the model along its weight dimensions: each linear layer's weight matrix is partitioned column-wise (output dimension) or row-wise (input dimension) across workers, and an all-reduce or all-gather collective synchronizes the partial outputs. For attention, the heads are partitioned across workers: each worker holds Q/K/V projections for its assigned subset of heads, computes attention independently, and an all-gather merges the head outputs. Tensor parallelism scales well to within-node GPU groups (NVLink bandwidth ~600 GB/s on H100) but degrades across nodes; pipeline parallelism is preferred for inter-node scaling.

Document 5 — Speculative decoding integration:
vLLM 0.5+ ships built-in speculative decoding via spec_decode_proposers. The most common configuration is K=4 or K=5 with a draft model running one or two sizes smaller than the target (e.g., Llama-3-8B drafting for Llama-3-70B). vLLM's spec decode runs draft and target in the same Python process; the draft proposes K tokens, the target verifies in one forward pass with prompt_logprobs=K to extract per-position acceptance, and rejected tokens are discarded with the bonus token from target's distribution sampled instead. Limitations: vLLM's built-in spec decode shares one GPU pool between draft and target, which works well for small draft models but doesn't scale to a TP=8 target with a TP=4 draft (the GPU accounting becomes ambiguous). Off-engine spec decode — running draft and target as separate vLLM instances, possibly on different nodes, with an external orchestrator — gives more flexibility but loses some efficiency.

Document 6 — vLLM benchmarking practices:
vLLM benchmarks should distinguish prefill throughput from decode throughput because they're bottlenecked by different resources. Prefill is compute-bound (FLOPS scale linearly with prefix tokens); decode is HBM-bandwidth-bound (each step reads all model weights). The standard benchmark suites (vLLM benchmark_serving.py, sglang's bench_serving.py, NVIDIA's GenAI-Perf) measure end-to-end latency under varying RPS and report TTFT (time to first token), TPOT (time per output token), and ITL (inter-token latency). For research benchmarks, a fixed batch size and fixed output length per cell (a 2D bs × N grid) is common because it pins down the compute regime. Care: gpu_memory_utilization affects the available KV cache size, which in turn affects effective batch size at long sequences — comparing benchmarks across mem-util settings is unsafe without normalizing.

Question: According to the context, what does PagedAttention's attention kernel do at the CUDA level (be specific), what does chunked prefill achieve in terms of head-of-line blocking, what is the specific limitation of vLLM's built-in speculative decoding when scaling target TP, and why is comparing benchmarks across gpu_memory_utilization settings unsafe? Answer in 8-10 sentences citing specific details from each relevant document.""",
]


# Curated prompt set spanning task types (concise to keep probe cheap)
CURATED_PROMPTS = [
    "Explain in one paragraph why the sky appears blue.",
    "Write a haiku about autumn.",
    "What's the capital of Australia?",
    "Translate 'good morning' into French, Spanish, and Japanese.",
    "Describe the difference between a stack and a queue in computer science.",
    "What's 17 times 23?",
    "Write a short tweet about why people love pizza.",
    "List three pros and three cons of remote work.",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "What is the boiling point of water in Celsius and Fahrenheit?",
    "Give a brief overview of how a bicycle works mechanically.",
    "Write a polite email declining a meeting invitation.",
    "What is the speed of light in vacuum?",
    "Compose a four-line poem about a cup of coffee.",
    "Explain photosynthesis to a 10-year-old.",
    "What does the acronym DNA stand for?",
    "Recommend a book for someone who enjoyed The Great Gatsby.",
    "Write a python one-liner to reverse a string.",
    "Describe the taste of dark chocolate.",
    "What year did World War II end?",
    "Give three tips for a beginner learning to play guitar.",
    "Write a one-paragraph horror story.",
    "Convert 100 USD to approximate EUR (just a rough estimate).",
    "What's the difference between weather and climate?",
    "Compose a short birthday message for a coworker.",
    "Name three planets in our solar system.",
    "Write a SQL query to select all rows from table 'users' where age > 18.",
    "Define 'metaphor' and give one example.",
    "What's the largest mammal on Earth?",
    "Write three lines of opening dialogue for a detective story.",
    "Explain what 'machine learning' means in one sentence.",
    "List five common house plants.",
    "Write a brief restaurant review (positive) for a sushi place.",
    "What is HTTP and what does it stand for?",
    "Give a one-sentence summary of the theory of relativity.",
    "Compose a thank-you note to a teacher.",
    "What's the difference between TCP and UDP?",
    "Write a tagline for a new electric scooter brand.",
    "Define 'inflation' in economics.",
    "Name five common allergens in food.",
    "Write a python function that returns the factorial of n.",
    "What's the chemical formula for water?",
    "Give two examples of a renewable energy source.",
    "Write a 50-word short story about a lost dog.",
    "What's the difference between an API and a library?",
    "Recommend a movie for a rainy Sunday afternoon.",
    "Define 'algorithm' in one sentence.",
    "What is the Pythagorean theorem?",
    "Write a polite request asking for a deadline extension.",
    "Name three Shakespeare plays.",
]


def parse_verdict(judge_text: str, draft_response: str) -> dict:
    """Parse the judge's verdict. Returns dict with 'verdict' and optional metadata."""
    text = judge_text.strip()

    # Take only the first line/verdict portion. Models sometimes ramble after.
    # Look for the first occurrence of any verdict keyword.
    if "REJECT" in text and (text.find("REJECT") < text.find("ACCEPT") if "ACCEPT" in text else True):
        if text.find("REJECT") < (text.find("TRUNCATE_AFTER") if "TRUNCATE_AFTER" in text else len(text)):
            return {"verdict": "REJECT", "raw": text[:200]}

    if text.startswith("ACCEPT") or "\nACCEPT" in text:
        return {"verdict": "ACCEPT", "raw": text[:200]}

    if "TRUNCATE_AFTER" in text:
        # Extract quoted substring after TRUNCATE_AFTER:
        # Pattern: TRUNCATE_AFTER: "..."
        idx = text.find("TRUNCATE_AFTER")
        after = text[idx:]
        # Find first quoted region
        first_q = after.find('"')
        if first_q == -1:
            return {"verdict": "TRUNCATE_PARSE_ERR", "raw": text[:200],
                    "error": "no quote after TRUNCATE_AFTER"}
        second_q = after.find('"', first_q + 1)
        if second_q == -1:
            return {"verdict": "TRUNCATE_PARSE_ERR", "raw": text[:200],
                    "error": "unmatched quote"}
        substring = after[first_q + 1:second_q]
        # Validate substring appears in draft response
        substring_idx = draft_response.find(substring)
        return {
            "verdict": "TRUNCATE_AFTER",
            "substring": substring,
            "substring_found_in_draft": substring_idx >= 0,
            "truncate_position_chars": substring_idx + len(substring) if substring_idx >= 0 else -1,
            "raw": text[:200],
        }

    return {"verdict": "PARSE_ERROR", "raw": text[:300]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--target", default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--draft-tp", type=int, default=4)
    ap.add_argument("--target-tp", type=int, default=8)
    ap.add_argument("--draft-mem-util", type=float, default=0.20)
    ap.add_argument("--target-mem-util", type=float, default=0.55)
    ap.add_argument("--target-gpu-blocks", type=int, default=4000)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--backend", default="mp")

    ap.add_argument("--n-prompts", type=int, default=50,
                    help="Number of curated prompts to evaluate")
    ap.add_argument("--draft-max-tokens", type=int, default=128,
                    help="Max tokens for the small model's response")
    ap.add_argument("--judge-max-tokens", type=int, default=80,
                    help="Max tokens for the judge's verdict output")
    ap.add_argument("--n-consistency-samples", type=int, default=20,
                    help="Re-run judge on this many samples to measure self-consistency")
    ap.add_argument("--mps-config", default="80_20")
    ap.add_argument("--output", default="/tmp/probe_judge.json")
    ap.add_argument("--samples-output", default="/tmp/judge_samples.json",
                    help="Save 20 random samples for human spot-check")
    args = ap.parse_args()

    print("=" * 78)
    print("Y2 CALIBRATION PROBE — does judge-verify make sense on real prompts?")
    print("=" * 78)
    print(f"  Draft: {args.draft}")
    print(f"  Target: {args.target}")
    print(f"  Prompts: {args.n_prompts} (from curated set)")
    print(f"  draft_max_tokens: {args.draft_max_tokens}")
    print(f"  judge_max_tokens: {args.judge_max_tokens}")
    print()

    print("[init] Connecting to Ray...")
    ray.init()

    # Kill any orphan vLLM workers from prior failed runs on the GPU node
    # before we try to allocate engines (idempotent).
    cleanup_vllm_workers()

    target_env: dict[str, str] = {}
    draft_env: dict[str, str] = {}
    if args.mps_config != "off":
        t_pct, d_pct = args.mps_config.split("_")
        target_env = {"CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps",
                      "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": t_pct}
        draft_env = {"CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps",
                     "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": d_pct}

    print("[init] Launching target...")
    target = TargetEngine.options(
        num_gpus=0, runtime_env={"env_vars": target_env},
    ).remote(
        model_id=args.target,
        tensor_parallel_size=args.target_tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.target_mem_util,
        distributed_executor_backend=args.backend,
        num_gpu_blocks_override=args.target_gpu_blocks,
    )
    t = time.perf_counter()
    ray.get(target.ping.remote())
    print(f"[init] Target loaded in {time.perf_counter() - t:.1f}s")

    print("[init] Launching draft...")
    draft = DraftEngine.options(
        num_gpus=0, runtime_env={"env_vars": draft_env},
    ).remote(
        model_id=args.draft,
        tensor_parallel_size=args.draft_tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.draft_mem_util,
        distributed_executor_backend=args.backend,
    )
    t = time.perf_counter()
    ray.get(draft.ping.remote())
    print(f"[init] Draft loaded in {time.perf_counter() - t:.1f}s")

    prompts = CURATED_PROMPTS[:args.n_prompts]

    # Step 1: draft generates responses
    print(f"\n[step 1] Drafting {len(prompts)} responses (max {args.draft_max_tokens} tokens each)...")
    t = time.perf_counter()
    draft_responses = ray.get(draft.generate_text.remote(
        prompts, max_tokens=args.draft_max_tokens,
    ))
    print(f"[step 1] Done in {time.perf_counter() - t:.1f}s. Sample draft:")
    print(f"  Q: {prompts[0]}")
    print(f"  A: {draft_responses[0][:200]}")

    # Step 2: target evaluates each via judge prompt
    print(f"\n[step 2] Target evaluating each draft via judge prompt...")
    judge_inputs = [
        VERIFY_PROMPT_TEMPLATE.format(query=p, response=r)
        for p, r in zip(prompts, draft_responses)
    ]
    t = time.perf_counter()
    judge_outputs = ray.get(target.generate_text.remote(
        judge_inputs, max_tokens=args.judge_max_tokens,
    ))
    print(f"[step 2] Done in {time.perf_counter() - t:.1f}s")

    # Step 3: parse verdicts
    print(f"\n[step 3] Parsing verdicts...")
    verdicts = []
    for i, (q, dr, jo) in enumerate(zip(prompts, draft_responses, judge_outputs)):
        v = parse_verdict(jo, dr)
        v["query"] = q
        v["draft_response"] = dr
        v["judge_raw"] = jo
        verdicts.append(v)

    # Stats
    counts = Counter(v["verdict"] for v in verdicts)
    n_total = len(verdicts)
    n_truncate = counts.get("TRUNCATE_AFTER", 0) + counts.get("TRUNCATE_PARSE_ERR", 0)
    n_truncate_clean = sum(1 for v in verdicts
                           if v["verdict"] == "TRUNCATE_AFTER"
                           and v.get("substring_found_in_draft", False))
    parse_err = counts.get("PARSE_ERROR", 0) + counts.get("TRUNCATE_PARSE_ERR", 0)
    overall_parse_success = (n_total - parse_err) / n_total

    print(f"\n=== Verdict distribution ===")
    for verdict, n in counts.most_common():
        print(f"  {verdict:25s} {n:>3} ({n / n_total:.0%})")
    print(f"\n=== TRUNCATE substring-match rate ===")
    if n_truncate > 0:
        print(f"  {n_truncate_clean}/{n_truncate} TRUNCATE verdicts had substring "
              f"that matched draft ({n_truncate_clean / n_truncate:.0%})")
    print(f"\n=== Overall parse success ===")
    print(f"  {n_total - parse_err}/{n_total} = {overall_parse_success:.0%}")

    # Step 4: self-consistency
    n_consist = min(args.n_consistency_samples, len(prompts))
    print(f"\n[step 4] Self-consistency on {n_consist} samples (re-run judge)...")
    sample_ids = sorted(random.sample(range(len(prompts)), n_consist))
    consist_inputs = [judge_inputs[i] for i in sample_ids]
    t = time.perf_counter()
    consist_outputs = ray.get(target.generate_text.remote(
        consist_inputs, max_tokens=args.judge_max_tokens,
    ))
    print(f"[step 4] Done in {time.perf_counter() - t:.1f}s")

    matched = 0
    for src_idx, jo in zip(sample_ids, consist_outputs):
        v_re = parse_verdict(jo, draft_responses[src_idx])
        if v_re["verdict"] == verdicts[src_idx]["verdict"]:
            matched += 1
    consistency = matched / n_consist
    print(f"\n=== Self-consistency ===")
    print(f"  {matched}/{n_consist} = {consistency:.0%}")

    # Step 5: save samples for human spot-check
    n_samples = min(20, len(verdicts))
    sample = random.sample(verdicts, n_samples)
    with open(args.samples_output, "w") as f:
        json.dump(sample, f, indent=2, default=str)
    print(f"\n[done] Saved {n_samples} random samples to {args.samples_output} for human spot-check")

    # Verdict on the probe itself
    print("\n" + "=" * 78)
    nondegenerate = (
        counts.get("ACCEPT", 0) > 0
        and (counts.get("TRUNCATE_AFTER", 0) > 0 or counts.get("REJECT", 0) > 0)
    )
    if overall_parse_success >= 0.90 and consistency >= 0.85 and nondegenerate:
        print("PROBE VERDICT: PASS — judge-verify is reliable enough to commit to Y2-C.")
        ret = 0
    elif overall_parse_success >= 0.80 and consistency >= 0.70 and nondegenerate:
        print("PROBE VERDICT: MARGINAL — usable but parser/prompt could be tightened.")
        ret = 0
    else:
        print("PROBE VERDICT: FAIL — refine the verify prompt template before Y2-C.")
        if overall_parse_success < 0.80:
            print(f"  Parse success {overall_parse_success:.0%} below 80% threshold")
        if consistency < 0.70:
            print(f"  Self-consistency {consistency:.0%} below 70% threshold")
        if not nondegenerate:
            print(f"  Verdict distribution is degenerate (likely all ACCEPT — sycophancy)")
        ret = 1
    print("=" * 78)

    # Save full results
    summary = {
        "args": vars(args),
        "verdict_counts": dict(counts),
        "n_total": n_total,
        "parse_success_rate": overall_parse_success,
        "truncate_substring_match_rate": (
            n_truncate_clean / n_truncate if n_truncate > 0 else None
        ),
        "self_consistency_rate": consistency,
        "verdicts": verdicts,
    }
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Full results: {args.output}")

    ray.shutdown()
    return ret


if __name__ == "__main__":
    sys.exit(main())
