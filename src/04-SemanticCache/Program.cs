// ============================================================
//  MAF Learning – 04: Semantic Cache Middleware
//
//  Instead of exact-match caching, we embed each query and
//  use cosine similarity to find semantically similar cached
//  responses — so paraphrased queries still hit the cache.
//
//  Flow:
//    Query arrives
//      → Embed query
//      → Cosine similarity check against cached embeddings
//      → Hit  : return cached response (no model call)
//      → Miss : call model, store embedding + response
// ============================================================

using Azure;
using Azure.AI.OpenAI;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;

// ── 1. Load .env and config ───────────────────────────────────
DotNetEnv.Env.Load(Path.Combine(AppContext.BaseDirectory, ".env"));

IConfiguration config = new ConfigurationBuilder()
    .SetBasePath(AppContext.BaseDirectory)
    .AddJsonFile("appsettings.json", optional: true)
    .AddEnvironmentVariables()
    .Build();

string endpoint            = config["AzureAI:Endpoint"]            ?? throw new InvalidOperationException("AzureAI:Endpoint missing");
string apiKey              = config["AzureAI:ApiKey"]              ?? throw new InvalidOperationException("AzureAI:ApiKey missing");
string modelDeployment     = config["AzureAI:ModelDeployment"]     ?? "o4-mini";
string embeddingDeployment = config["AzureAI:EmbeddingDeployment"] ?? "text-embedding-ada-002";

Console.WriteLine("==============================================");
Console.WriteLine("  MAF Learning – 04: Semantic Cache");
Console.WriteLine("==============================================");
Console.WriteLine($"  Endpoint  : {endpoint}");
Console.WriteLine($"  Model     : {modelDeployment}");
Console.WriteLine($"  Embedding : {embeddingDeployment}");
Console.WriteLine();

// ── 2. Create Azure OpenAI client ─────────────────────────────
AzureOpenAIClient azureClient = new(
    new Uri(endpoint),
    new AzureKeyCredential(apiKey),
    new AzureOpenAIClientOptions(AzureOpenAIClientOptions.ServiceVersion.V2024_12_01_Preview)
);

// ── 3. Create embedding generator ────────────────────────────
//  Used by SemanticCache to convert queries into float vectors.
IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator = azureClient
    .GetEmbeddingClient(embeddingDeployment)
    .AsIEmbeddingGenerator();

// ── 4. Create semantic cache ──────────────────────────────────
//  threshold = 0.95 → catches near-identical and paraphrased queries
//  Lower it (e.g. 0.85) to catch broader topic matches
SemanticCache semanticCache = new(embeddingGenerator, threshold: 0.88f);

// ── 5. Build IChatClient with semantic cache middleware ───────
//  The .Use() middleware intercepts every GetResponseAsync call:
//    - Embeds the last user message
//    - Checks SemanticCache for a similar query
//    - Returns cached response on hit, otherwise calls the model
//      and stores the result for future queries
IChatClient chatClient = azureClient
    .GetChatClient(modelDeployment)
    .AsIChatClient()
    .AsBuilder()
    .Use(
        async (messages, options, next, ct) =>
        {
            // Extract the last user message as the lookup key
            string query = messages.LastOrDefault(m => m.Role == ChatRole.User)?.Text
                           ?? string.Empty;

            // Check semantic cache
            string? cached = await semanticCache.FindAsync(query, ct);

            if (cached is not null)
            {
                Console.WriteLine("  [SemanticCache] HIT — returning cached response");
                return new ChatResponse([new ChatMessage(ChatRole.Assistant, cached)]);
            }

            // Cache miss — call the model
            Console.WriteLine("  [SemanticCache] MISS — calling model");
            ChatResponse response = await next.GetResponseAsync(messages, options, ct);

            // Store for future similar queries
            if (response.Text is not null)
                await semanticCache.StoreAsync(query, response.Text, ct);

            return response;
        },
        null
    )
    .Build();

// ── 6. Create agent ───────────────────────────────────────────
ChatClientAgent agent = new ChatClientAgent(
    chatClient: chatClient,
    name: "SemanticCacheAgent",
    instructions: "You are a helpful AI assistant. Answer clearly and concisely."
);

// ── 7. Run queries ────────────────────────────────────────────
//  Query 1 — original (cache miss, model called)
//  Query 2 — paraphrase (cache hit, model NOT called)
//  Query 3 — different topic (cache miss, model called)

var queries = new[]
{
    ("1st", "Explain how transformers work in machine learning"),
    ("2nd", "How does the transformer architecture work in ML?"),          // paraphrase → should hit
    ("3rd", "What is the best way to cook pasta?"),                       // different topic → miss
};

foreach (var (label, query) in queries)
{
    AgentSession session = await agent.CreateSessionAsync();

    Console.WriteLine($"[{label}] User: {query}");
    AgentResponse response = await agent.RunAsync(message: query, session: session);
    Console.WriteLine($"[{label}] Agent: {response.Text}");
    Console.WriteLine();
}

Console.WriteLine("Done!");
