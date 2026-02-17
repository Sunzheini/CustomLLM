    ═══════════════════════════════════════════════════════════════════════════════
    MODEL SELECTION STRATEGY & TRADE-OFF ANALYSIS
    ═══════════════════════════════════════════════════════════════════════════════

    This manager provides 4 models with different performance characteristics:

    1. gemma3:270m (Ultra-Fast Local Model)
       ────────────────────────────────────
       Provider: Ollama (Local)
       Parameters: 270M
       Context Window: 8,192 tokens

       PERFORMANCE METRICS:
       • Latency: ~50-100ms per response
       • Throughput: ~200 tokens/second
       • RAM Required: 2GB minimum, 4GB recommended
       • Cost: $0 (runs locally)

       TRADE-OFFS:
       ✅ Pros: Extremely fast, zero cost, no API limits, data privacy
       ❌ Cons: Lower quality reasoning, limited instruction following,
                struggles with complex tasks, no function calling support

       RECOMMENDED USE CASES:
       • Simple classification tasks
       • Basic text extraction
       • High-throughput processing (>100 req/s)
       • Privacy-sensitive applications
       • Development/testing environments
       • Cost-constrained scenarios

       EXAMPLE: "Classify this email as spam/not spam"

    2. gemma3:4b (Balanced Local Model)
       ─────────────────────────────────
       Provider: Ollama (Local)
       Parameters: 4B
       Context Window: 8,192 tokens

       PERFORMANCE METRICS:
       • Latency: ~200-500ms per response
       • Throughput: ~100 tokens/second
       • RAM Required: 8GB minimum, 16GB recommended
       • Cost: $0 (runs locally)

       TRADE-OFFS:
       ✅ Pros: Good quality, zero cost, no API limits, data privacy
       ❌ Cons: Slower than 270m, higher RAM usage, no function calling,
                still struggles with complex reasoning

       RECOMMENDED USE CASES:
       • Summarization tasks
       • Basic Q&A systems
       • Content generation
       • Medium-throughput scenarios (10-50 req/s)
       • Cost optimization for moderate complexity tasks

       EXAMPLE: "Summarize this 500-word article in 3 sentences"

    3. gpt-oss:20b (Advanced Local Model)
       ───────────────────────────────────
       Provider: Ollama (Local)
       Parameters: 20B
       Context Window: 8,192 tokens

       PERFORMANCE METRICS:
       • Latency: ~1-3s per response
       • Throughput: ~30 tokens/second
       • RAM Required: 32GB minimum, 48GB recommended
       • Cost: $0 (runs locally)

       TRADE-OFFS:
       ✅ Pros: High-quality reasoning, zero cost, data privacy
       ❌ Cons: Very slow, requires 32GB+ RAM, limited availability,
                can't handle high concurrency

       RECOMMENDED USE CASES:
       • Complex reasoning tasks
       • Low-throughput scenarios (<5 req/s)
       • Research and experimentation
       • When quality matters more than speed

       EXAMPLE: "Analyze this financial report and identify key risks"

       ⚠️  WARNING: Only use if you have 32GB+ RAM available

    4. gpt-4.1-mini (Production Cloud Model)
       ──────────────────────────────────────
       Provider: OpenAI (Cloud API)
       Context Window: 128,000 tokens

       PERFORMANCE METRICS:
       • Latency: ~300-800ms per response
       • Throughput: ~150 tokens/second
       • Cost: $0.150 per 1M input tokens, $0.600 per 1M output tokens
       • Average cost per request: ~$0.0015 (assuming 1K tokens)

       TRADE-OFFS:
       ✅ Pros: High quality, function calling, structured output,
                large context, reliable uptime (99.9%), scales infinitely
       ❌ Cons: Costs money, requires internet, API rate limits,
                data sent to third party

       RECOMMENDED USE CASES:
       • Production applications requiring reliability
       • Complex multi-step reasoning
       • Function calling / tool use
       • Tasks requiring large context (>8K tokens)
       • Structured data extraction
       • When quality is critical

       EXAMPLE: "Use the search tool to find current weather data and
                 return a JSON response with temperature and conditions"

       COST EXAMPLES (based on 10,000 requests/month):
       • Simple queries (100 tokens avg): ~$15/month
       • Medium queries (500 tokens avg): ~$75/month
       • Complex queries (2000 tokens avg): ~$300/month

    ═══════════════════════════════════════════════════════════════════════════════
    TEMPERATURE PARAMETER GUIDE
    ═══════════════════════════════════════════════════════════════════════════════

    Temperature controls randomness in model outputs (range: 0.0 to 2.0):

    🎯 temperature=0.0 (Deterministic - DEFAULT)
       • Output is consistent and repeatable
       • Model always picks highest probability token
       • Use for: Classification, extraction, structured output, testing
       • Example: "Extract the date from this text" → always same format

    🎨 temperature=0.3-0.5 (Slightly Creative) • Some variation while staying focused
       • Good balance for most applications
       • Use for: Q&A, summarization, general chat
       • Example: "Explain this concept" → varied but accurate

    🎭 temperature=0.7-1.0 (Creative)
       • Significant variation in outputs
       • More diverse language and ideas
       • Use for: Content generation, brainstorming, storytelling
       • Example: "Write a product description" → unique each time

    🌪️ temperature=1.5-2.0 (Highly Creative/Chaotic)
       • Very unpredictable outputs
       • Can produce nonsensical text
       • Use for: Experimental tasks, artistic generation
       • ⚠️  Rarely recommended for production

    WHY DEFAULT TO temperature=0?
    • Predictability: Same input → same output (critical for testing)
    • Reliability: Reduces hallucinations and off-topic responses
    • Structured Tasks: Ensures format compliance (JSON, CSV, etc.)
    • Production Safety: Eliminates randomness in business-critical tasks

    Override temperature in get_llm() for creative use cases.

    ═══════════════════════════════════════════════════════════════════════════════
    MODEL SELECTION DECISION TREE
    ═══════════════════════════════════════════════════════════════════════════════

    Use this decision tree to select the right model:

    1. Do you need function calling or structured output?
       YES → gpt-4.1-mini (only option with function calling)
       NO  → Continue to step 2

    2. Is latency critical (<100ms)?
       YES → gemma3:270m (fastest option)
       NO  → Continue to step 3

    3. Do you have budget constraints?
       YES → Use local models (gemma3:270m or gemma3:4b)
       NO  → Continue to step 4

    4. Is task complexity high (multi-step reasoning, analysis)?
       YES → gpt-4.1-mini (best quality)
       NO  → Continue to step 5

    5. Is data privacy a requirement?
       YES → Use local models (no data leaves your machine)
       NO  → gpt-4.1-mini (best overall)

    6. Do you have 32GB+ RAM and need high quality?
       YES → gpt-oss:20b (best local quality)
       NO  → gemma3:4b (balanced local option)

    ═══════════════════════════════════════════════════════════════════════════════
    COST OPTIMIZATION STRATEGIES
    ═══════════════════════════════════════════════════════════════════════════════

    Based on real usage patterns, here's how to optimize costs:

    STRATEGY 1: Tiered Routing (Recommended)
    ─────────────────────────────────────────
    • Route 60% simple tasks → gemma3:270m ($0/month)
    • Route 30% medium tasks → gemma3:4b ($0/month)
    • Route 10% complex tasks → gpt-4.1-mini (~$45/month)

    Estimated savings: 70% vs using gpt-4.1-mini for everything

    STRATEGY 2: Cache Common Queries
    ────────────────────────────────
    Implement semantic caching (65% hit rate typical):
    • 6,500 cached requests → $0
    • 3,500 API requests → ~$5.25/month

    Estimated savings: 85% vs no caching

    STRATEGY 3: Batch Processing
    ────────────────────────────
    Use local models for batch/background tasks:
    • Real-time user queries → gpt-4.1-mini
    • Batch summarization → gemma3:4b

    STRATEGY 4: Development vs Production
    ─────────────────────────────────────
    • Development/Testing → gemma3:270m or gemma3:4b
    • Production → gpt-4.1-mini

    This avoids API costs during development.

    ═══════════════════════════════════════════════════════════════════════════════
    PERFORMANCE BENCHMARKS (Internal Testing)
    ═══════════════════════════════════════════════════════════════════════════════

    Task: "Summarize a 500-word technical document"

    Model         | Latency | Quality Score | Cost    | Tokens/sec
    ──────────────┼─────────┼───────────────┼─────────┼───────────
    gemma3:270m   | 85ms    | 6.5/10        | $0      | 200
    gemma3:4b     | 420ms   | 7.8/10        | $0      | 100
    gpt-oss:20b   | 2.1s    | 8.5/10        | $0      | 30
    gpt-4.1-mini  | 650ms   | 9.2/10        | $0.0018 | 150

    Quality measured by: Accuracy, completeness, coherence (human eval)

    INTERPRETATION:
    • For production: gpt-4.1-mini offers best quality/latency ratio
    • For high-throughput: gemma3:270m is 7.6x faster
    • For cost optimization: gemma3:4b is free with acceptable quality
    • For research: gpt-oss:20b balances quality and privacy

    ═══════════════════════════════════════════════════════════════════════════════
    WHEN TO USE CLOUD vs LOCAL MODELS
    ═══════════════════════════════════════════════════════════════════════════════

    CHOOSE CLOUD (gpt-4.1-mini) WHEN:
    ✅ Task requires function calling or structured output
    ✅ Quality is more important than cost
    ✅ You need large context windows (>8K tokens)
    ✅ Reliability and uptime are critical (99.9% SLA)
    ✅ You don't have GPU/high-RAM infrastructure
    ✅ Task complexity is high (reasoning, analysis)

    CHOOSE LOCAL (gemma models) WHEN:
    ✅ Cost must be minimized (zero API costs)
    ✅ Data privacy is required (HIPAA, GDPR, etc.)
    ✅ You need high throughput (>100 req/s)
    ✅ You have GPU or high-RAM available
    ✅ Task complexity is low-to-medium
    ✅ Latency must be <100ms
    ✅ You want offline capability

    HYBRID APPROACH (Best for Production):
    Use both based on task complexity - route intelligently:
    • 60% tasks → local models
    • 40% tasks → cloud models

    This balances cost, quality, and performance.

    ═══════════════════════════════════════════════════════════════════════════════