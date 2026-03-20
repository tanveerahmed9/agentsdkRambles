// ============================================================
//  MAF Learning – 05: Redis Cache + Azure AI Search
//
//  Hybrid semantic cache:
//    Azure AI Search → stores query embeddings, similarity lookup
//    Azure Cache for Redis → stores response text, fast retrieval
//
//  Flow:
//    1. Incoming query → embed with ada-002
//    2. Vector search in AI Search for nearest cached query
//    3. Hit  (score ≥ threshold) → fetch response from Redis
//    4. Miss → call model → store in Redis → index in AI Search
//
//  Required environment variables (.env):
//    AzureAI__Endpoint, AzureAI__ApiKey
//    AzureAI__ModelDeployment, AzureAI__EmbeddingDeployment
//    AzureSearch__Endpoint, AzureSearch__ApiKey, AzureSearch__IndexName
//    Redis__ConnectionString
// ============================================================

using Azure;
using Azure.AI.OpenAI;
using Azure.Search.Documents;
using Azure.Search.Documents.Indexes;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;
using StackExchange.Redis;

// ── 1. Load .env and config ───────────────────────────────────
// Load .env from the output directory (where it is copied on build)
string envPath = Path.Combine(AppContext.BaseDirectory, ".env");
DotNetEnv.Env.Load(envPath);

IConfiguration config = new ConfigurationBuilder()
    .SetBasePath(AppContext.BaseDirectory)
    .AddJsonFile("appsettings.json", optional: true)
    .AddEnvironmentVariables()
    .Build();

string aiEndpoint            = config["AzureAI:Endpoint"]             ?? throw new InvalidOperationException("AzureAI:Endpoint missing");
string aiApiKey              = config["AzureAI:ApiKey"]               ?? throw new InvalidOperationException("AzureAI:ApiKey missing");
string modelDeployment       = config["AzureAI:ModelDeployment"]      ?? "o4-mini";
string embeddingDeployment   = config["AzureAI:EmbeddingDeployment"]  ?? "text-embedding-ada-002";
string searchEndpoint        = config["AzureSearch:Endpoint"]         ?? throw new InvalidOperationException("AzureSearch:Endpoint missing");
string searchApiKey          = config["AzureSearch:ApiKey"]           ?? throw new InvalidOperationException("AzureSearch:ApiKey missing");
string searchIndexName       = config["AzureSearch:IndexName"]        ?? "semantic-cache";
string redisConnectionString = config["Redis:ConnectionString"]       ?? throw new InvalidOperationException("Redis:ConnectionString missing");

// Azure ML is optional — only used when endpoint is configured
string? azureMlEndpoint    = config["AzureML:EmbeddingEndpoint"];
string? azureMlApiKey      = config["AzureML:ApiKey"];
int embeddingDimensions    = int.TryParse(config["AzureML:EmbeddingDimensions"], out int dims) ? dims : 1536;

Console.WriteLine("==============================================");
Console.WriteLine("  MAF Learning – 05: Redis + AI Search Cache");
Console.WriteLine("==============================================");
Console.WriteLine($"  Model     : {modelDeployment}");
Console.WriteLine($"  AI Search : {searchIndexName}");

// ── 2. Azure OpenAI client ────────────────────────────────────
AzureOpenAIClient azureClient = new(
    new Uri(aiEndpoint),
    new AzureKeyCredential(aiApiKey),
    new AzureOpenAIClientOptions(AzureOpenAIClientOptions.ServiceVersion.V2024_12_01_Preview)
);

// ── 3. Embedding generator ────────────────────────────────────
//  If AzureML:EmbeddingEndpoint is configured, use the domain-specific
//  Azure ML model (e.g. energy fine-tuned sentence-transformer).
//  Otherwise fall back to Azure OpenAI embeddings (ada-002).
IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator;

if (!string.IsNullOrWhiteSpace(azureMlEndpoint) && !string.IsNullOrWhiteSpace(azureMlApiKey))
{
    Console.WriteLine($"  Embedding : Azure ML (domain-specific) — dims: {embeddingDimensions}");
    embeddingGenerator = new AzureMLEmbeddingGenerator(azureMlEndpoint, azureMlApiKey, embeddingDimensions);
}
else
{
    Console.WriteLine($"  Embedding : Azure OpenAI ({embeddingDeployment}) — dims: {embeddingDimensions}");
    embeddingGenerator = azureClient
        .GetEmbeddingClient(embeddingDeployment)
        .AsIEmbeddingGenerator();
}

Console.WriteLine();

// ── 4. Azure AI Search clients ────────────────────────────────
//  SearchIndexClient → manages index schema (create/delete)
//  SearchClient      → queries and uploads documents
var searchCredential = new AzureKeyCredential(searchApiKey);

SearchIndexClient indexClient = new(new Uri(searchEndpoint), searchCredential);
SearchClient searchClient     = new(new Uri(searchEndpoint), searchIndexName, searchCredential);

// ── 5. Redis client ───────────────────────────────────────────
//  ConnectionMultiplexer is thread-safe and should be reused.
ConnectionMultiplexer redisConnection = await ConnectionMultiplexer.ConnectAsync(redisConnectionString);
IDatabase redis = redisConnection.GetDatabase();

// ── 6. Hybrid cache ───────────────────────────────────────────
HybridCache hybridCache = new(
    searchClient,
    indexClient,
    redis,
    embeddingGenerator,
    searchIndexName,
    threshold: 0.88f,
    embeddingDimensions: embeddingDimensions
);

// Create AI Search index if it doesn't exist
await hybridCache.EnsureIndexExistsAsync();
Console.WriteLine();

// ── 7. Build IChatClient with hybrid cache middleware ─────────
IChatClient chatClient = azureClient
    .GetChatClient(modelDeployment)
    .AsIChatClient()
    .AsBuilder()
    .Use(
        async (messages, options, next, ct) =>
        {
            string query = messages.LastOrDefault(m => m.Role == ChatRole.User)?.Text
                           ?? string.Empty;

            // Try semantic cache first
            string? cached = await hybridCache.FindAsync(query, ct);

            if (cached is not null)
            {
                Console.WriteLine("  [Cache] HIT — returning cached response\n");
                return new ChatResponse([new ChatMessage(ChatRole.Assistant, cached)]);
            }

            // Miss — call the model
            Console.WriteLine("  [Cache] MISS — calling model");
            ChatResponse response = await next.GetResponseAsync(messages, options, ct);

            // Store in hybrid cache
            if (response.Text is not null)
                await hybridCache.StoreAsync(query, response.Text, ct);

            Console.WriteLine();
            return response;
        },
        null
    )
    .Build();

// ── 8. Create agent ───────────────────────────────────────────
ChatClientAgent agent = new ChatClientAgent(
    chatClient: chatClient,
    name: "HybridCacheAgent",
    instructions: "You are a helpful AI assistant. Answer clearly and concisely."
);

// ── 9. Run test queries ───────────────────────────────────────
//  Query 1 — first run, cache miss, model called
//  Query 2 — paraphrase of query 1, should hit cache
//  Query 3 — different topic, miss

var queries = new[]
{
    // ── Seed the cache with a neural network transformer query
    ("1st", "How does the transformer architecture work in neural networks?"),

    // ── Paraphrase of the same ML transformer → should HIT
    ("2nd", "Explain the self-attention mechanism used in large language models"),

    // ── Electrical transformer — completely different domain, same word
    // With general embeddings this might incorrectly score high against the ML entry
    ("3rd", "How does an electrical transformer step up and step down voltage?"),

    // ── Another electrical transformer question — should NOT hit the ML cache
    ("4th", "What causes transformer core losses in a power distribution system?"),
};

foreach (var (label, query) in queries)
{
    AgentSession session = await agent.CreateSessionAsync();

    Console.WriteLine($"[{label}] User : {query}");
    AgentResponse response = await agent.RunAsync(message: query, session: session);
    Console.WriteLine($"[{label}] Agent: {response.Text}");
    Console.WriteLine(new string('-', 60));
    Console.WriteLine();
}

redisConnection.Dispose();
Console.WriteLine("Done!");
